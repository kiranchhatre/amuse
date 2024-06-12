
import torch
import torch.nn as nn
from einops import rearrange
import numpy as np
from scipy import stats
from sklearn import metrics
import torchmetrics as tmetrics

from models.audio.audio_main_new import ASTModel

class FusionBlock(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=2):
        super(FusionBlock, self).__init__()
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=input_dim, nhead=4) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(input_dim)
        self.fc = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.fc(x)
        return x
    
class DecoderBlock(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=4):
        super(DecoderBlock, self).__init__()
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=input_dim, nhead=4) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(input_dim)
        self.projection = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, output_dim)
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.projection(x)
        return x

class AST_EVP(nn.Module):
    
    def __init__(self):
        super(AST_EVP, self).__init__()
        
        input_dim = 256
        latent_dim = 512
        output_dim = 1024 * 128
        
        self.emo_enc = ASTModel(label_dim=8, fstride=10, tstride=10, input_fdim=128,
                                input_tdim=1024, imagenet_pretrain=True,
                                audioset_pretrain=False, model_size='base384', verbose=False)
        self.sty_enc = ASTModel(label_dim=30, fstride=10, tstride=10, input_fdim=128,
                                input_tdim=1024, imagenet_pretrain=True,
                                audioset_pretrain=False, model_size='base384', verbose=False)
        self.con_enc = ASTModel(label_dim=0, fstride=10, tstride=10, input_fdim=128,
                                input_tdim=1024, imagenet_pretrain=True,
                                audioset_pretrain=False, model_size='base384', verbose=False)
        
        self.fusion = FusionBlock(input_dim * 3, latent_dim)
        self.fusion_ablation = FusionBlock(input_dim * 2, latent_dim)
        self.decode = DecoderBlock(latent_dim, output_dim)
        
        self.l1_loss = nn.L1Loss()
        self.CrossEntropyLoss = nn.CrossEntropyLoss()
        
    def reconstruct(self, x, reconstruct_only=False, frame_based_feats=False):
    # def forward(self, x):
        if not reconstruct_only:
            emo = self.emo_enc(x, frame_based_feats)
            sty = self.sty_enc(x, frame_based_feats)
            con = self.con_enc(x, frame_based_feats)
            x = torch.cat((emo['feature'], sty['feature'], con['feature']), dim=-1)  # Shape: batchsize, 256 * 3, sequence: emo, sty, con
        else: emo, sty, con = None, None, None
        
        latent = self.fusion(x)  # Shape: batchsize, latent_dim
        fbanks = self.decode(latent)  # Shape: batchsize, output_dim
        fbanks = rearrange(fbanks, 'b (h w) -> b h w', h=1024)
        return fbanks, emo, sty, con
    
    def eval_func(self, x, frame_based_feats=False, metrics=False):
        if not metrics:
            with torch.no_grad():
                emo, sty, con = self.emo_enc(x, frame_based_feats), self.sty_enc(x, frame_based_feats), self.con_enc(x, frame_based_feats)
            return {"emo": emo["feature"].squeeze(0), 
                    "sty": sty["feature"].squeeze(0), 
                    "con": con["feature"].squeeze(0)}
        else:
            with torch.no_grad():
                fbanks, emo, sty, con = self.reconstruct(x, reconstruct_only=False, frame_based_feats=True)
                recons_emo, recons_sty, recons_con = self.emo_enc(fbanks, frame_based_feats), self.sty_enc(fbanks, frame_based_feats), self.con_enc(fbanks, frame_based_feats)
            return {
                "fbanks": fbanks.squeeze(0),
                "emo": {"feature": emo["feature"].squeeze(0), "predicted_labels": emo["predicted_labels"]},
                "sty": {"feature": sty["feature"].squeeze(0), "predicted_labels": sty["predicted_labels"]},
                "con": {"feature": con["feature"].squeeze(0), "predicted_labels": con["predicted_labels"]},
                "new_emo": {"feature": recons_emo["feature"].squeeze(0), "predicted_labels": recons_emo["predicted_labels"]},
                "new_sty": {"feature": recons_sty["feature"].squeeze(0), "predicted_labels": recons_sty["predicted_labels"]},
                "new_con": {"feature": recons_con["feature"].squeeze(0), "predicted_labels": recons_con["predicted_labels"]},
            }
    
    def reconstruct_ablation(self, x, reconstruct_only=False, ablation="", frame_based_feats=False):
        if not reconstruct_only:
            con = self.con_enc(x)
            if ablation == "emotion": bb = self.emo_enc(x, frame_based_feats)
            elif ablation == "identity": bb = self.sty_enc(x, frame_based_feats)
            x = torch.cat((bb['feature'], con['feature']), dim=-1)  # Shape: batchsize, 256 * 2, sequence: sty, con
        else: bb, con = None, None
    
        latent = self.fusion_ablation(x)  # Shape: batchsize, latent_dim
        fbanks = self.decode(latent)  # Shape: batchsize, output_dim
        fbanks = rearrange(fbanks, 'b (h w) -> b h w', h=1024)
        return fbanks, bb, con

    def forward(self, data, noise, ablation="full", frame_based_feats=False):
        
        if ablation in ["emotion", "identity"]: return self.forward_ablation(data, noise, ablation, frame_based_feats)
        elif ablation in ["ast_baseline"]: return self.forward_baseline(data, noise, ablation, frame_based_feats)
        else: assert ablation in ["full"], f"[AST MODEL] expected emotion identity ast_baseline or full, but got {ablation}"
        
        emo_id, a1_id, a2_id = data["emo_id"], data["a1_id"], data["a2_id"]
        if not noise: a1_t1, a1_t2, a2_t1, a2_t2 = data["fbank_a1_t1"], data["fbank_a1_t2"], data["fbank_a2_t1"], data["fbank_a2_t2"]
        else: a1_t1, a1_t2, a2_t1, a2_t2 = data["fbank_a1_t1_noisy"], data["fbank_a1_t2_noisy"], data["fbank_a2_t1_noisy"], data["fbank_a2_t2_noisy"]
        
        # reconstructions
        reconstruct_only = False
        a1_t1_dict, a1_t2_dict, a2_t1_dict, a2_t2_dict = {}, {}, {}, {}
        
        # self reconstructions
        a1_t1_dict["fbanks_self_a1_t1"], a1_t1_dict["emo"], a1_t1_dict["sty"], a1_t1_dict["con"] = self.reconstruct(a1_t1, reconstruct_only, frame_based_feats)
        a1_t2_dict["fbanks_self_a1_t2"], a1_t2_dict["emo"], a1_t2_dict["sty"], a1_t2_dict["con"] = self.reconstruct(a1_t2, reconstruct_only, frame_based_feats)
        a2_t1_dict["fbanks_self_a2_t1"], a2_t1_dict["emo"], a2_t1_dict["sty"], a2_t1_dict["con"] = self.reconstruct(a2_t1, reconstruct_only, frame_based_feats)
        a2_t2_dict["fbanks_self_a2_t2"], a2_t2_dict["emo"], a2_t2_dict["sty"], a2_t2_dict["con"] = self.reconstruct(a2_t2, reconstruct_only, frame_based_feats)
        
        # cross reconstructions
        reconstruct_only = True
        
        # content swaps
        a1_t1_dict["fbanks_c_a2_t1"], _, _, _ = self.reconstruct(torch.cat((a1_t1_dict["emo"]['feature'], a1_t1_dict["sty"]['feature'], a2_t1_dict["con"]['feature']), dim=-1), reconstruct_only)
        a1_t2_dict["fbanks_c_a2_t2"], _, _, _ = self.reconstruct(torch.cat((a1_t2_dict["emo"]['feature'], a1_t2_dict["sty"]['feature'], a2_t2_dict["con"]['feature']), dim=-1), reconstruct_only)
        a2_t1_dict["fbanks_c_a1_t1"], _, _, _ = self.reconstruct(torch.cat((a2_t1_dict["emo"]['feature'], a2_t1_dict["sty"]['feature'], a1_t1_dict["con"]['feature']), dim=-1), reconstruct_only)
        a2_t2_dict["fbanks_c_a1_t2"], _, _, _ = self.reconstruct(torch.cat((a2_t2_dict["emo"]['feature'], a2_t2_dict["sty"]['feature'], a1_t2_dict["con"]['feature']), dim=-1), reconstruct_only)
        
        # emotion swaps
        a1_t1_dict["fbanks_e_a1_t2"], _, _, _ = self.reconstruct(torch.cat((a1_t2_dict["emo"]['feature'], a1_t1_dict["sty"]['feature'], a1_t1_dict["con"]['feature']), dim=-1), reconstruct_only)
        a1_t2_dict["fbanks_e_a1_t1"], _, _, _ = self.reconstruct(torch.cat((a1_t1_dict["emo"]['feature'], a1_t2_dict["sty"]['feature'], a1_t2_dict["con"]['feature']), dim=-1), reconstruct_only)
        a2_t1_dict["fbanks_e_a2_t2"], _, _, _ = self.reconstruct(torch.cat((a2_t2_dict["emo"]['feature'], a2_t1_dict["sty"]['feature'], a2_t1_dict["con"]['feature']), dim=-1), reconstruct_only)
        a2_t2_dict["fbanks_e_a2_t1"], _, _, _ = self.reconstruct(torch.cat((a2_t1_dict["emo"]['feature'], a2_t2_dict["sty"]['feature'], a2_t2_dict["con"]['feature']), dim=-1), reconstruct_only)
        
        # style swaps
        a1_t1_dict["fbanks_s_a1_t2"], _, _, _ = self.reconstruct(torch.cat((a1_t1_dict["emo"]['feature'], a1_t2_dict["sty"]['feature'], a1_t1_dict["con"]['feature']), dim=-1), reconstruct_only)
        a1_t2_dict["fbanks_s_a1_t1"], _, _, _ = self.reconstruct(torch.cat((a1_t2_dict["emo"]['feature'], a1_t1_dict["sty"]['feature'], a1_t2_dict["con"]['feature']), dim=-1), reconstruct_only)
        a2_t1_dict["fbanks_s_a2_t2"], _, _, _ = self.reconstruct(torch.cat((a2_t1_dict["emo"]['feature'], a2_t2_dict["sty"]['feature'], a2_t1_dict["con"]['feature']), dim=-1), reconstruct_only)
        a2_t2_dict["fbanks_s_a2_t1"], _, _, _ = self.reconstruct(torch.cat((a2_t2_dict["emo"]['feature'], a2_t1_dict["sty"]['feature'], a2_t2_dict["con"]['feature']), dim=-1), reconstruct_only)
        
        if noise: a1_t1, a1_t2, a2_t1, a2_t2 = data["fbank_a1_t1"], data["fbank_a1_t2"], data["fbank_a2_t1"], data["fbank_a2_t2"]
        elememt_list = [[a1_t1_dict, a1_t1, "a1t1"], [a1_t2_dict, a1_t2, "a1t2"], 
                        [a2_t1_dict, a2_t1, "a2t1"], [a2_t2_dict, a2_t2, "a2t2"]]
        total_loss, emo_acc, person_id_acc, loss_dict = self._collect_metrics(elememt_list, emo_id, a1_id, a2_id)
        return {"loss": total_loss, "emo_acc": emo_acc, "person_id_acc": person_id_acc, 
                "a1_t1_emo": a1_t1_dict["emo"]['predicted_labels'], "a1_t2_emo": a1_t2_dict["emo"]['predicted_labels'],
                "a2_t1_emo": a2_t1_dict["emo"]['predicted_labels'], "a2_t2_emo": a2_t2_dict["emo"]['predicted_labels'],
                "a1_t1_sty": a1_t1_dict["sty"]['predicted_labels'], "a1_t2_sty": a1_t2_dict["sty"]['predicted_labels'],
                "a2_t1_sty": a2_t1_dict["sty"]['predicted_labels'], "a2_t2_sty": a2_t2_dict["sty"]['predicted_labels'],
                "a1_id": a1_id, "a2_id": a2_id, "emo_id": emo_id, "loss_dict": loss_dict}

    def forward_baseline(self, data, noise, ablation, frame_based_feats=False):
        assert ablation in ["ast_baseline"], f"[AST MODEL] ablation should be ast_baseline, but got {ablation}"
        # assert not noise, f"[AST MODEL BASELINE] noise should be False, but got {noise}"
        feat_name = "emo"
        
        emo_id, a1_id, a2_id = data["emo_id"], data["a1_id"], data["a2_id"]
        if not noise: a1_t1, a1_t2, a2_t1, a2_t2 = data["fbank_a1_t1"], data["fbank_a1_t2"], data["fbank_a2_t1"], data["fbank_a2_t2"]
        else: a1_t1, a1_t2, a2_t1, a2_t2 = data["fbank_a1_t1_noisy"], data["fbank_a1_t2_noisy"], data["fbank_a2_t1_noisy"], data["fbank_a2_t2_noisy"]
        
        # reconstructions
        reconstruct_only = False
        a1_t1_dict, a1_t2_dict, a2_t1_dict, a2_t2_dict = {}, {}, {}, {}
        
        # self reconstructions
        a1_t1_dict["fbanks_self_a1_t1"], a1_t1_dict[feat_name], a1_t1_dict["con"] = self.reconstruct_ablation(a1_t1, reconstruct_only, "emotion", frame_based_feats)
        a1_t2_dict["fbanks_self_a1_t2"], a1_t2_dict[feat_name], a1_t2_dict["con"] = self.reconstruct_ablation(a1_t2, reconstruct_only, "emotion", frame_based_feats)
        a2_t1_dict["fbanks_self_a2_t1"], a2_t1_dict[feat_name], a2_t1_dict["con"] = self.reconstruct_ablation(a2_t1, reconstruct_only, "emotion", frame_based_feats)
        a2_t2_dict["fbanks_self_a2_t2"], a2_t2_dict[feat_name], a2_t2_dict["con"] = self.reconstruct_ablation(a2_t2, reconstruct_only, "emotion", frame_based_feats)
        
        # cross reconstructions
        reconstruct_only = True
        
        # EVP-based swaps
        a1_t1_dict["fbanks_c_a2_t1"], _, _ = self.reconstruct_ablation(torch.cat((a1_t1_dict[feat_name]['feature'], a2_t1_dict["con"]['feature']), dim=-1), reconstruct_only, "emotion")
        a1_t2_dict["fbanks_c_a2_t2"], _, _ = self.reconstruct_ablation(torch.cat((a1_t2_dict[feat_name]['feature'], a2_t2_dict["con"]['feature']), dim=-1), reconstruct_only, "emotion")
        a2_t1_dict["fbanks_c_a1_t1"], _, _ = self.reconstruct_ablation(torch.cat((a2_t1_dict[feat_name]['feature'], a1_t1_dict["con"]['feature']), dim=-1), reconstruct_only, "emotion")
        a2_t2_dict["fbanks_c_a1_t2"], _, _ = self.reconstruct_ablation(torch.cat((a2_t2_dict[feat_name]['feature'], a1_t2_dict["con"]['feature']), dim=-1), reconstruct_only, "emotion")
        
        if noise: a1_t1, a1_t2, a2_t1, a2_t2 = data["fbank_a1_t1"], data["fbank_a1_t2"], data["fbank_a2_t1"], data["fbank_a2_t2"]
        elememt_list = [[a1_t1_dict, a1_t1, "a1t1"], [a1_t2_dict, a1_t2, "a1t2"],
                        [a2_t1_dict, a2_t1, "a2t1"], [a2_t2_dict, a2_t2, "a2t2"]]
        total_loss, emo_acc, person_id_acc, loss_dict = self._collect_metrics(elememt_list, emo_id, a1_id, a2_id, ablation=ablation)
        return {"loss": total_loss, "emo_acc": emo_acc, "person_id_acc": person_id_acc,      
                "a1_t1_emo": a1_t1_dict[feat_name]['predicted_labels'], "a1_t2_emo": a1_t2_dict[feat_name]['predicted_labels'],
                "a2_t1_emo": a2_t1_dict[feat_name]['predicted_labels'], "a2_t2_emo": a2_t2_dict[feat_name]['predicted_labels'],
                "a1_t1_sty": None, "a1_t2_sty": None,
                "a2_t1_sty": None, "a2_t2_sty": None,
                "a1_id": a1_id, "a2_id": a2_id, "emo_id": emo_id, "loss_dict": loss_dict}

    def forward_ablation(self, data, noise, ablation, frame_based_feats=False):
        assert ablation in ["emotion", "identity"], f"[AST MODEL] ablation should be either emotion or identity, but got {ablation}"
        feat_name = "emo" if ablation == "emotion" else "sty" if ablation == "identity" else print(int(f"[AST-T ABLATION] {ablation} is not supported"))
        
        emo_id, a1_id, a2_id = data["emo_id"], data["a1_id"], data["a2_id"]
        if not noise: a1_t1, a1_t2, a2_t1, a2_t2 = data["fbank_a1_t1"], data["fbank_a1_t2"], data["fbank_a2_t1"], data["fbank_a2_t2"]
        else: a1_t1, a1_t2, a2_t1, a2_t2 = data["fbank_a1_t1_noisy"], data["fbank_a1_t2_noisy"], data["fbank_a2_t1_noisy"], data["fbank_a2_t2_noisy"]
        
        # reconstructions
        reconstruct_only = False
        a1_t1_dict, a1_t2_dict, a2_t1_dict, a2_t2_dict = {}, {}, {}, {}
        
        # self reconstructions
        a1_t1_dict["fbanks_self_a1_t1"], a1_t1_dict[feat_name], a1_t1_dict["con"] = self.reconstruct_ablation(a1_t1, reconstruct_only, ablation, frame_based_feats)
        a1_t2_dict["fbanks_self_a1_t2"], a1_t2_dict[feat_name], a1_t2_dict["con"] = self.reconstruct_ablation(a1_t2, reconstruct_only, ablation, frame_based_feats)
        a2_t1_dict["fbanks_self_a2_t1"], a2_t1_dict[feat_name], a2_t1_dict["con"] = self.reconstruct_ablation(a2_t1, reconstruct_only, ablation, frame_based_feats)
        a2_t2_dict["fbanks_self_a2_t2"], a2_t2_dict[feat_name], a2_t2_dict["con"] = self.reconstruct_ablation(a2_t2, reconstruct_only, ablation, frame_based_feats)
        
        # cross reconstructions
        reconstruct_only = True
        
        # content swaps
        a1_t1_dict["fbanks_c_a2_t1"], _, _ = self.reconstruct_ablation(torch.cat((a1_t1_dict[feat_name]['feature'], a2_t1_dict["con"]['feature']), dim=-1), reconstruct_only, ablation)
        a1_t2_dict["fbanks_c_a2_t2"], _, _ = self.reconstruct_ablation(torch.cat((a1_t2_dict[feat_name]['feature'], a2_t2_dict["con"]['feature']), dim=-1), reconstruct_only, ablation)
        a2_t1_dict["fbanks_c_a1_t1"], _, _ = self.reconstruct_ablation(torch.cat((a2_t1_dict[feat_name]['feature'], a1_t1_dict["con"]['feature']), dim=-1), reconstruct_only, ablation)
        a2_t2_dict["fbanks_c_a1_t2"], _, _ = self.reconstruct_ablation(torch.cat((a2_t2_dict[feat_name]['feature'], a1_t2_dict["con"]['feature']), dim=-1), reconstruct_only, ablation)
        
        # emotion/ style swaps
        a1_t1_dict["fbanks_e_a1_t2"], _, _ = self.reconstruct_ablation(torch.cat((a1_t2_dict[feat_name]['feature'], a1_t1_dict["con"]['feature']), dim=-1), reconstruct_only, ablation)
        a1_t2_dict["fbanks_e_a1_t1"], _, _ = self.reconstruct_ablation(torch.cat((a1_t1_dict[feat_name]['feature'], a1_t2_dict["con"]['feature']), dim=-1), reconstruct_only, ablation)
        a2_t1_dict["fbanks_e_a2_t2"], _, _ = self.reconstruct_ablation(torch.cat((a2_t2_dict[feat_name]['feature'], a2_t1_dict["con"]['feature']), dim=-1), reconstruct_only, ablation)
        a2_t2_dict["fbanks_e_a2_t1"], _, _ = self.reconstruct_ablation(torch.cat((a2_t1_dict[feat_name]['feature'], a2_t2_dict["con"]['feature']), dim=-1), reconstruct_only, ablation)
        
        if noise: a1_t1, a1_t2, a2_t1, a2_t2 = data["fbank_a1_t1"], data["fbank_a1_t2"], data["fbank_a2_t1"], data["fbank_a2_t2"]
        elememt_list = [[a1_t1_dict, a1_t1, "a1t1"], [a1_t2_dict, a1_t2, "a1t2"],
                        [a2_t1_dict, a2_t1, "a2t1"], [a2_t2_dict, a2_t2, "a2t2"]]
        total_loss, emo_acc, person_id_acc, loss_dict = self._collect_metrics(elememt_list, emo_id, a1_id, a2_id, ablation=ablation)
        if ablation == "emotion":
            ablation_based_dict = {"a1_t1_emo": a1_t1_dict[feat_name]['predicted_labels'], "a1_t2_emo": a1_t2_dict[feat_name]['predicted_labels'],
                                   "a2_t1_emo": a2_t1_dict[feat_name]['predicted_labels'], "a2_t2_emo": a2_t2_dict[feat_name]['predicted_labels'],
                                   "a1_t1_sty": None, "a1_t2_sty": None,
                                   "a2_t1_sty": None, "a2_t2_sty": None,}
        elif ablation == "identity":   
            ablation_based_dict = {"a1_t1_emo": None, "a1_t2_emo": None,
                                   "a2_t1_emo": None, "a2_t2_emo": None,
                                   "a1_t1_sty": a1_t1_dict[feat_name]['predicted_labels'], "a1_t2_sty": a1_t2_dict[feat_name]['predicted_labels'],
                                   "a2_t1_sty": a2_t1_dict[feat_name]['predicted_labels'], "a2_t2_sty": a2_t2_dict[feat_name]['predicted_labels'],}
        base_dict = {"loss": total_loss, "emo_acc": emo_acc, "person_id_acc": person_id_acc,      
                     "a1_id": a1_id, "a2_id": a2_id, "emo_id": emo_id, "loss_dict": loss_dict}
        return {**base_dict, **ablation_based_dict}

    def loss_n_acc(self, pred, tgt, loss_type):
        if loss_type == "ce":
            return self.CrossEntropyLoss(pred, tgt)
        elif loss_type == "acc":
            _, pred = pred.topk(1, 1)
            pred0 = pred.squeeze().data 
            return 100 * torch.sum(pred0 == tgt.data) / tgt.size(0)
        elif loss_type == "l1":
            return self.l1_loss(pred, tgt)
        elif loss_type == "focal":
            raise NotImplementedError("Focal loss is not implemented yet")   
 
    def _collect_metrics(self, elememt_list, emo_id, a1_id, a2_id, ablation="full"):
        
        ablation_key = "emo" if ablation in ["emotion", "ast_baseline"] \
                  else "sty" if ablation == "identity"                  \
                  else "emo_sty" if ablation == "full"                  \
                  else print(int(f"[AST-T METRICS] {ablation} is not supported"))
                  
        loss_dict = {}
        total_loss, emo_acc, person_id_acc = 0.0, 0.0, 0.0
        for element in elememt_list:
            dicts = element[0]
            gt_list = [element[1], element[2]]
            for k in dicts.keys():
                if "fbanks" in k: 
                    LOSS = self.loss_n_acc(dicts[k], gt_list[0], "l1")
                    loss_dict[f"{k}_{element[2]}"] = LOSS
                    total_loss += LOSS
                if k in ablation_key and k == "emo":
                    LOSS = self.loss_n_acc(dicts[k]['predicted_labels'], emo_id, "ce")
                    loss_dict[f"{k}_{element[2]}"] = LOSS
                    total_loss += LOSS
                    ACC = self.loss_n_acc(dicts[k]['predicted_labels'], emo_id, "acc")
                    loss_dict[f"{k}_{element[2]}_acc"] = ACC
                    emo_acc += ACC
                if k in ablation_key and k == "sty":
                    if gt_list[1] in ["a1t1", "a1t2"]:
                        LOSS = self.loss_n_acc(dicts[k]['predicted_labels'], a1_id, "ce")
                        loss_dict[f"{k}_{element[2]}"] = LOSS
                        total_loss += LOSS
                        ACC = self.loss_n_acc(dicts[k]['predicted_labels'], a1_id, "acc")
                        loss_dict[f"{k}_{element[2]}_acc"] = ACC
                        person_id_acc += ACC
                    elif gt_list[1] in ["a2t1", "a2t2"]:
                        LOSS = self.loss_n_acc(dicts[k]['predicted_labels'], a2_id, "ce")
                        loss_dict[f"{k}_{element[2]}"] = LOSS
                        total_loss += LOSS
                        ACC = self.loss_n_acc(dicts[k]['predicted_labels'], a2_id, "acc")
                        loss_dict[f"{k}_{element[2]}_acc"] = ACC
                        person_id_acc += ACC
                    else: raise ValueError(f"gt_list[1] should be either a1 or a2, but got {gt_list[1]}")
        
        # content alignment supervision
        LOSS = self.loss_n_acc(elememt_list[0][0]["con"]['feature'], elememt_list[2][0]["con"]['feature'], "l1")
        loss_dict["con_a1_a2_t1"] = LOSS
        total_loss += LOSS
        LOSS = self.loss_n_acc(elememt_list[1][0]["con"]['feature'], elememt_list[3][0]["con"]['feature'], "l1")
        loss_dict["con_a1_a2_t2"] = LOSS
        total_loss += LOSS
        
        # averaging accuracies
        emo_acc /= 4
        person_id_acc /= 4
        
        return total_loss, emo_acc, person_id_acc, loss_dict
    
    def d_prime(self, auc): # Not used
        standard_normal = stats.norm()
        return standard_normal.ppf(auc) * np.sqrt(2)   
        
    def calculate_stats(self, outputs, parallelism=False, ablation="full"):
        
        a1_t1_emo = torch.cat([x["a1_t1_emo"].unsqueeze(0).cpu().detach() for x in outputs], dim=0) if ablation not in ["identity"] else None
        a1_t2_emo = torch.cat([x["a1_t2_emo"].unsqueeze(0).cpu().detach() for x in outputs], dim=0) if ablation not in ["identity"] else None
        a2_t1_emo = torch.cat([x["a2_t1_emo"].unsqueeze(0).cpu().detach() for x in outputs], dim=0) if ablation not in ["identity"] else None
        a2_t2_emo = torch.cat([x["a2_t2_emo"].unsqueeze(0).cpu().detach() for x in outputs], dim=0) if ablation not in ["identity"] else None
        a1_t1_sty = torch.cat([x["a1_t1_sty"].unsqueeze(0).cpu().detach() for x in outputs], dim=0) if ablation not in ["emotion", "ast_baseline"] else None
        a1_t2_sty = torch.cat([x["a1_t2_sty"].unsqueeze(0).cpu().detach() for x in outputs], dim=0) if ablation not in ["emotion", "ast_baseline"] else None
        a2_t1_sty = torch.cat([x["a2_t1_sty"].unsqueeze(0).cpu().detach() for x in outputs], dim=0) if ablation not in ["emotion", "ast_baseline"] else None
        a2_t2_sty = torch.cat([x["a2_t2_sty"].unsqueeze(0).cpu().detach() for x in outputs], dim=0) if ablation not in ["emotion", "ast_baseline"] else None
        a1_id = torch.cat([x["a1_id"].unsqueeze(0).cpu().detach() for x in outputs], dim=0)
        a2_id = torch.cat([x["a2_id"].unsqueeze(0).cpu().detach() for x in outputs], dim=0)
        emo_id = torch.cat([x["emo_id"].unsqueeze(0).cpu().detach() for x in outputs], dim=0)
        
        if ablation not in ["identity"]:
            combined_emo = torch.cat([a1_t1_emo, a1_t2_emo, a2_t1_emo, a2_t2_emo], dim=0)
            if parallelism:
                combined_emo  = rearrange(combined_emo, 'b t d -> (b t) d')
                combined_emo = combined_emo.unsqueeze(1)
            combined_emo = rearrange(combined_emo, 'b t d -> (b t) d')
            if not parallelism: _, combined_emo = combined_emo.topk(1,1)
            else: 
                combined_emo = combined_emo.to(dtype=torch.float32)
                _, combined_emo = combined_emo.topk(1,1)
                combined_emo = combined_emo.to(dtype=torch.int64)
            combined_emo = combined_emo.squeeze()
            combined_emo_labels = torch.cat([emo_id, emo_id, emo_id, emo_id], dim=0)
            combined_emo_labels = rearrange(combined_emo_labels, 'b t -> (b t)')
        
        if ablation not in ["emotion", "ast_baseline"]:
            combined_subjects = torch.cat([a1_t1_sty, a1_t2_sty, a2_t1_sty, a2_t2_sty], dim=0)
            if parallelism:
                combined_subjects  = rearrange(combined_subjects, 'b t d -> (b t) d')
                combined_subjects = combined_subjects.unsqueeze(1)
            combined_subjects = rearrange(combined_subjects, 'b t d -> (b t) d')
            if not parallelism: _, combined_subjects = combined_subjects.topk(1,1)
            else:
                combined_subjects = combined_subjects.to(dtype=torch.float32)
                _, combined_subjects = combined_subjects.topk(1,1)
                combined_subjects = combined_subjects.to(dtype=torch.int64)
            combined_subjects = combined_subjects.squeeze()
            combined_subjects_labels = torch.cat([a1_id, a1_id, a2_id, a2_id], dim=0)
            combined_subjects_labels = rearrange(combined_subjects_labels, 'b t -> (b t)')
        
        # stats
        stat_datas = [(combined_emo, combined_emo_labels, 8), (combined_subjects, combined_subjects_labels, 30)] if ablation == "full" \
                        else [(combined_emo, combined_emo_labels, 8)] if ablation in ["emotion", "ast_baseline"] \
                        else [(combined_subjects, combined_subjects_labels, 30)] if ablation == "identity" \
                        else print(int(f"[AST-T STATS] {ablation} is not supported"))
        for stat_data in stat_datas:
            combined, combined_labels, n_classes = stat_data
            
            precision = tmetrics.Precision(average='macro', num_classes=n_classes)
            f1 = tmetrics.F1Score(num_classes=n_classes)
            recall = tmetrics.Recall(average='macro', num_classes=n_classes)
            
            results = {
                "acc": tmetrics.functional.accuracy(combined, combined_labels) * 100,
                "average_precisions": precision(combined, combined_labels),
                "f1": f1(combined, combined_labels),
                "recall": recall(combined, combined_labels)
            }
            
            if n_classes == 8: emo_stats_results = results
            elif n_classes == 30: subject_stats_results = results            
            
            # Deprecated AST version, use torchmetrics instead
            ##
            # for k in range(combined.shape[1]):
            #     avg_precision = metrics.average_precision_score(combined_labels[:, k], combined[:, k], average=None)
            #     auc = metrics.roc_auc_score(combined_labels[:, k], combined[:, k], average=None)
            #     (precisions, recalls, thresholds) = metrics.precision_recall_curve(combined_labels[:, k], combined[:, k])
            #     (fpr, tpr, thresholds) = metrics.roc_curve(combined_labels[:, k], combined[:, k])
            #     # sample stats to reduce size
            #     idx = np.random.choice(len(precisions), size=100, replace=False)
            #     precisions = precisions[idx]
            #     recalls = recalls[idx]
            #     thresholds = thresholds[idx]
            #     idx = np.random.choice(len(fpr), size=100, replace=False)
            #     fpr = fpr[idx]
            #     fnr = 1 - tpr[idx]
            #     results.append({
            #         "AP": avg_precision, 
            #         "auc": auc, 
            #         "precisions": precisions, 
            #         "recalls": recalls, 
            #         "fpr": fpr, 
            #         "fnr": fnr
            #     })
            # results["mAP"] = np.mean([x["AP"] for x in results])
            # results["mAUC"] = np.mean([x["auc"] for x in results])
            # middle_ps = [x["precisions"][int(len(x["precisions"]) / 2)] for x in results]
            # middle_rs = [x["recalls"][int(len(x["recalls"]) / 2)] for x in results]
            # results["average_precisions"] = np.mean(middle_ps)
            # results["average_recalls"] = np.mean(middle_rs)
            # results["d_prime"] = self.d_prime(results["mAUC"])
        
        if ablation == "full": return {"emo_stats": emo_stats_results, "subject_stats": subject_stats_results}
        elif ablation in ["emotion", "ast_baseline"]: return {"emo_stats": emo_stats_results, "subject_stats": None}
        elif ablation == "identity": return {"subject_stats": subject_stats_results, "emo_stats": None}
                    
if __name__ == '__main__':
    
    # Usage example
    
    from audio_main_new import ASTModel
    
    datatype = torch.float32
    model = AST_EVP().to(dtype=datatype, device="cuda")
    for p in model.parameters(): p.data = p.data.to(datatype)
    params_in_mil = sum(p.numel() for p in model.parameters()) / 1e6
    print('total params: ', params_in_mil, 'M')
    x = torch.randn(4, 1024, 128).to(dtype=datatype, device="cuda")
    x = x.to(datatype)
    fbanks, emo, sty, con = model(x)
    print("fbanks", fbanks.shape)
    for k in emo.keys(): print(k, emo[k].shape)
    for k in sty.keys(): print(k, sty[k].shape)
    for k in con.keys(): 
        if con[k] is not None: print(k, con[k].shape)
