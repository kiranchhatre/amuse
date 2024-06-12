
import torch
import numpy as np
from pathlib import Path

class PretrainedAudioAE():
    
    def __init__(self, device) -> None:
        self.device = device
    
    def load_models(self, config, processed, tag, base_con_ae, base_emo_ae, backup_cfg):
        self.tag = tag
        self.config = config
        assert self.tag == "latent_diffusion", f"[LATDIFF] pretrained audio ae is only supported for latent diffusion, not for {self.tag}"
        
        con_ae_model_path = processed.parents[1] / "saved-models" / config["TRAIN_PARAM"][tag]["pretrained_audiocon"]
        if backup_cfg is not None: con_ae_model_path = Path(backup_cfg["pretrained_audiocon"])
        emo_ae_model_path = processed.parents[1] / "saved-models" / config["TRAIN_PARAM"][tag]["pretrained_audioemo"]
        if backup_cfg is not None: emo_ae_model_path = Path(backup_cfg["pretrained_audioemo"])
        con_ae_models = [f for f in con_ae_model_path.iterdir() if f.is_file() and "experiment_args.json" not in str(f)]
        emo_ae_models = [f for f in emo_ae_model_path.iterdir() if f.is_file() and "experiment_args.json" not in str(f)]

        # con AE
        tLoss, vLoss = np.inf, np.inf
        epoch = 0
        for con_ae_model in con_ae_models:
            model_id = str(con_ae_model).split("/")[-1].split(".pkl")[0]
            e, tL, vL = self._get_num(model_id.split("_")[1]), self._get_num(model_id.split("_")[2]), self._get_num(model_id.split("_")[3])
            if e > epoch:
                epoch = e
                tLoss, vLoss = tL, vL
                best_con_ae_model = con_ae_model
        print("[LATDIFF] <===== Chosen con AE model based on epoch: ", best_con_ae_model, " =====>") 
        self.base_con_ae = base_con_ae
        self.base_con_ae.to(self.device)
        self.base_con_ae.load_state_dict(torch.load(best_con_ae_model))
        for p in self.base_con_ae.parameters(): p.requires_grad = False
        self.base_con_ae.eval()
        
        # emo AE
        tRec, tEmo, tTotal, vRec, vEmo, vTotal = np.inf, np.inf, np.inf, np.inf, np.inf, np.inf
        for emo_ae_model in emo_ae_models:
            model_id = str(emo_ae_model).split("/")[-1].split(".pkl")[0]
            tR, tE, tT, vR, vE, vT = self._get_num(model_id.split("_")[1]), self._get_num(model_id.split("_")[2]), self._get_num(model_id.split("_")[3]), \
                                     self._get_num(model_id.split("_")[4]), self._get_num(model_id.split("_")[5]), self._get_num(model_id.split("_")[6])
            if tT < tTotal:
                tRec, tEmo, tTotal, vRec, vEmo, vTotal = tR, tE, tT, vR, vE, vT
                best_emo_ae_model = emo_ae_model
        print("[LATDIFF] <===== Chosen emo AE model based on total loss: ", best_emo_ae_model, " =====>")
        self.base_emo_ae = base_emo_ae
        self.base_emo_ae.to(self.device)
        self.base_emo_ae.load_state_dict(torch.load(best_emo_ae_model))
        for p in self.base_emo_ae.parameters(): p.requires_grad = False
        self.base_emo_ae.eval()
        
        return self.base_con_ae, self.base_emo_ae
    
    def _get_num(self, x):
        return int(''.join(ele for ele in x if ele.isdigit()))
    
class PretrainedBaseAudioAE():
    
    def __init__(self, device) -> None:
        self.device = device
        
    def load_model(self, config, processed, tag, base_ae, backup_cfg):
        self.tag = tag
        self.config = config
        assert self.tag == "latent_diffusion", f"[LATDIFF] pretrained audio ae is only supported for latent diffusion, not for {self.tag}"
        
        base_ae_model_path = processed.parents[1] / "saved-models" / config["TRAIN_PARAM"][tag]["pretained_baseaudio"]
        if backup_cfg is not None: base_ae_model_path = Path(backup_cfg["pretrained_baseae"])
        base_ae_models = [f for f in base_ae_model_path.iterdir() if f.is_file() and "experiment_args.json" not in str(f)]

        tLoss = np.inf
        for base_ae_model in base_ae_models:
            model_id = str(base_ae_model).split("/")[-1].split(".pt")[0]
            tL, e = self._get_num(model_id.split("_")[3]), self._get_num(model_id.split("_")[4])
            if tL < tLoss:
                tLoss = tL
                best_base_model = base_ae_model
        print("[LATDIFF] <===== Chosen BASE AE model based on total loss: ", best_base_model, " =====>")
        self.base_ae = base_ae
        self.base_ae.to(self.device)
        self.base_ae.load_state_dict(torch.load(best_base_model))
        for p in self.base_ae.parameters(): p.requires_grad = False
        self.base_ae.eval()
               
        return self.base_ae
    
    def _get_num(self, x):
        return int(''.join(ele for ele in x if ele.isdigit()))