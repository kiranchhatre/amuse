
import time
import torch
import numpy as np
import torch.utils.data as d
from torch.utils.data import DataLoader

from dm.utils.wav_utils import *

class PretrainedDTW():
    
    def __init__(self, device=None) -> None:
        if device is None: self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else: self.device = device
    
    def get_model(self, config, processed, tag):
        
        self.tag = tag
        emocla_tag="wav_mfcc"
        dtw_tag="wav_dtw_mfcc"

        saved_model_path = processed.parents[1] / "saved-models" / config["TRAIN_PARAM"][tag]["pretrained_dtw"]
        dtw_models = [f for f in saved_model_path.iterdir() if f.is_file() and "experiment_args.json" not in str(f)]

        tLoss, vLoss, tAcc, vAcc = 999999999, 999999999, 0, 0
        for dtw_model in dtw_models:
            model_id = str(dtw_model).split("/")[-1].split(".")[0] # 'model_27_tL9_tA69_vL25_vA17'
            e, tL, tA, vL, vA = self._get_num(model_id.split("_")[1]), \
                                self._get_num(model_id.split("_")[2]), \
                                self._get_num(model_id.split("_")[3]), \
                                self._get_num(model_id.split("_")[4]), \
                                self._get_num(model_id.split("_")[5])
            if tL < tLoss:
                tLoss, vLoss, tAcc, vAcc = tL, vL, tA, vA
                best_dtw_model = dtw_model
        print("[DIFF] (2/3) <===== Chosen DTW model based on train loss: ", best_dtw_model, " =====>")
        
        from models import allmodels # import here to avoid circular import
        self.model = allmodels[dtw_tag]
        self.model = self.model.to(self.device)
        self.model.load_state_dict(torch.load(best_dtw_model))
        self.model.eval()
        
        return self.model
    
    def get_features(self, wav_file, mfcc_transform, config, save_wf=False, ldm_eval=False):
        
        con, emo = [], []
        
        if not ldm_eval:
            actor = wav_file.split("/")[-1].split("_")[1]
            take = "_".join(wav_file.split("/")[-1].split("_")[2:]).split(".")[0]
        else: actor, take = "XXX", "0_X_X"
        
        sample_feat = {}
        sample_feat[actor] = {}
        sample_feat[actor][take] = {}

        if self.tag == "latent_diffusion":
            sample_feat = audio2mfcc(mfcc_transform, config, sample_feat, stage=self.tag, wav_file=wav_file, ldm_eval=ldm_eval)
            sample_mfcc = sample_feat[actor][take][self.tag]
        else:
            sample_feat = audio2mfcc(mfcc_transform, config, sample_feat, stage="wav_mfcc", wav_file=wav_file, save_wf=save_wf) # wav_mfcc stage to prevent concatenation
            sample_mfcc = sample_feat[actor][take]["wav_mfcc"]
            if save_wf: raw_wf = sample_feat[actor][take]["raw_wf"]

        test_set = test_dataloader(sample_mfcc)
        # test_loader = DataLoader(test_set, batch_size=self._largest_batchsize(sample_mfcc), shuffle=False, drop_last=True) # shuffle and drop_last must be False
        test_loader = DataLoader(test_set, batch_size=2, shuffle=False, drop_last=True) # latent_diffusion, bs=1 not functioning 
        
        for i, data in enumerate(test_loader):
            for k, v in data.items():
                data[k] = v.to(self.device) if k != "combo" else v
            reconstruct_dict, _, _ = self.model.val_func(data, triplet=None)
            con.append(reconstruct_dict["c1_1"].detach().cpu().numpy())
            emo.append(reconstruct_dict["e1_1"].detach().cpu().numpy())

        con = np.concatenate(con, axis=0) # (seq_len, 256)
        emo = np.concatenate(emo, axis=0) # (seq_len, 128)
        if save_wf: return {"con": con, "emo": emo, "raw_wf": raw_wf}
        else: return {"con": con, "emo": emo}
            
    def _get_num(self, x):
        return int(''.join(ele for ele in x if ele.isdigit()))

    def _largest_batchsize(self, sample_mfcc):
        power_of_two = [2**i for i in range(1, 20)] 
        samples = 0
        for bs in power_of_two:
            allin = bs * (len(sample_mfcc) // bs) 
            if allin > samples:
                samples = allin
                largest_batch_size = bs
        return largest_batch_size
    
class test_dataloader(d.Dataset):
    
    def __init__(self, sample_mfcc):
        self.sample_mfcc = sample_mfcc           

    def __len__(self):
        return len(self.sample_mfcc)
    
    def __getitem__(self, idx):
        
        sample_clip = self.sample_mfcc[idx]
        sample_clip = torch.FloatTensor(sample_clip)
        sample_clip = sample_clip[ :, 1:, :]
        dummy_clip = torch.zeros(sample_clip.shape)
        
        return {"combo": "",              # dummy
                "label": torch.tensor(0), # dummy
                "a1_t1": sample_clip, 
                "a1_t2": dummy_clip,      # dummy
                "a2_t1": dummy_clip,      # dummy
                "a2_t2": dummy_clip}      # dummy