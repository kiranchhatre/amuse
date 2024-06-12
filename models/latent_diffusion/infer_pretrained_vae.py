
import re
import torch
import numpy as np

from dm.utils.bvh_utils import *

class PretrainedVAE():
    
    def __init__(self, device) -> None:
        self.device = device
    
    def load_model(self, config, processed, tag, vae_base_arch, backup_cfg, lpdm_cfg=None):
        self.tag = tag
        self.config = config
        self.vae_base_arch = vae_base_arch
        self.seq_len = self.config["DATA_PARAM"]["Bvh"]["train_pose_framelen"]
        
        if lpdm_cfg is None:
            saved_model_path = processed.parents[1] / "saved-models" / config["TRAIN_PARAM"][tag]["pretrained_prior"]
            if backup_cfg is not None: saved_model_path = Path(backup_cfg["pretrained_prior"])
            vae_models = [f for f in saved_model_path.iterdir() if f.is_file() and "experiment_args.json" not in str(f)]
            rec_feat_L, rec_jts_L, kl_L, total_L = np.inf, np.inf, np.inf, np.inf
            for vae_model in vae_models:
                model_id = str(vae_model).split("/")[-1].split(".pt")[0]
                fL, jL, kL, tL, e = self._get_num(model_id.split("_")[1]), \
                                    self._get_num(model_id.split("_")[2]), self._get_num(model_id.split("_")[3]), \
                                    self._get_num(model_id.split("_")[4]), \
                                    self._get_num(model_id.split("_")[5])
                if tL < total_L:
                    rec_feat_L, rec_jts_L, kl_L, total_L = fL, jL, kL, tL
                    best_vae_model = vae_model
        else: 
            saved_model_path = processed.parents[1] / lpdm_cfg["saved_model_dir"] / config["TRAIN_PARAM"][tag]["pretrained_lpdm"]
            if backup_cfg is not None: raise NotImplementedError("Backup for LPDM not implemented!")
            vae_models = [f for f in saved_model_path.iterdir() if f.is_file() and "experiment_args.json" not in str(f) and f.stem.split("_")[0] == "prior"]
            if lpdm_cfg["load_epoch_prior"] == "best":
                total_L = np.inf
                for vae_model in vae_models:
                    tL = float(re.findall("\d+\.\d+", vae_model.stem.split("_")[-2])[0])
                    if tL < total_L: total_L = tL; best_vae_model = vae_model
            else: best_vae_model = [f for f in vae_models if int(re.search(r'\d+', f.stem.split("_")[-1]).group()) == int(lpdm_cfg["load_epoch_prior"])][0]
        
        print("[LATDIFF] <===== Chosen VAE model based on total loss: ", best_vae_model, " =====>") # "epoch", "model_state_dict", "optimizer_state_dict"
        chkpt = torch.load(best_vae_model)
        self.vae_base_arch.to(self.device)
        self.vae_base_arch.load_state_dict(chkpt["model_state_dict"])
        for p in self.vae_base_arch.parameters(): p.requires_grad = False
        self.vae_base_arch.eval()
    
    def get_latent(self, motion_batch):
        motion_batch.to(self.device) if not motion_batch.is_cuda else motion_batch
        lengths = [self.seq_len] * motion_batch.shape[0]
        with torch.no_grad():
            motion_z, _ = self.vae_base_arch.encode(features=motion_batch, lengths=lengths)
        return motion_z
    
    def get_motion(self, latent_batch, lengths):
        latent_batch.to(self.device) if not latent_batch.is_cuda else latent_batch
        with torch.no_grad():
            feats_rst = self.vae_base_arch.decode(z=latent_batch, lengths=lengths)
        return feats_rst
    
    def _get_num(self, x):
        return int(''.join(ele for ele in x if ele.isdigit()))