
import json
import torch
import numpy as np
import torch.nn as nn
from pathlib import Path
from torchmetrics import MetricCollection
from diffusers import DDIMScheduler, DDPMScheduler

from models.diffusion.bvh_fac.bvh_fac_models import PoseMLD, PoseMDM
from models.diffusion.denoiser import Denoiser
from models.diffusion.diffusion_losses import get_loss_fn

class DiffusionModel(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()  
    
    def setup(self, config, processed):
        self.config = config
        self.processed = processed
        self.tag = self.config["TRAIN_PARAM"]["tag"]
        self.bs = self.config["TRAIN_PARAM"][self.tag]["batch_size"]
        self.gpt = self.config["DATA_PARAM"]["Txtgrid"]["hf_model"]
        self.diffusion_type = self.config["TRAIN_PARAM"][self.tag]["arch"]
        self.use_diffusion = self.config["TRAIN_PARAM"][self.tag]["diff_mdm"]["use"]
        self.train_modality = self.config["TRAIN_PARAM"][self.tag]["con_emo_div"]["train_modality"]
        assert self.train_modality in ["bvh", "face", "both"], "Train modality should be either bvh, face or both"
        with open(str(Path(self.processed.parents[1], "configs/", self.diffusion_type + ".json"))) as f:
            self.diffusion_cfg = json.load(f)
            
        # Disentangled representation
        self.bvh_div = self.config["TRAIN_PARAM"][self.tag]["con_emo_div"]["bvh"]
        self.fac_div = self.config["TRAIN_PARAM"][self.tag]["con_emo_div"]["face"]
        assert self.bvh_div == self.fac_div, "Motion representations (BVH + FACE) should be of same kind: disentangled or not"
        
        if self.diffusion_type in ["diff_raw_camn", "diff_raw_pymo_camn", "diff_raw_mdm"]:
            # # Posenet
            if self.train_modality in ["bvh", "both"]:
                if self.bvh_div:
                    self.pos_con = PoseMDM(self.diffusion_cfg, self.gpt, self.processed, self.bs, arch_type="content", modality="pose")
                    self.pos_emo = PoseMDM(self.diffusion_cfg, self.gpt, self.processed, self.bs, arch_type="emotion", modality="pose")
                else:
                    self.pos_comb = PoseMDM(self.diffusion_cfg, self.gpt, self.processed, self.bs, arch_type="combo", modality="pose")
                
            # # FaceNet
            if self.train_modality in ["face", "both"]:
                if self.fac_div:
                    self.fac_con = PoseMDM(self.diffusion_cfg, self.gpt, self.processed, self.bs, arch_type="content", modality="face")
                    self.fac_emo = PoseMDM(self.diffusion_cfg, self.gpt, self.processed, self.bs, arch_type="emotion", modality="face")
                else:
                    self.fac_comb = PoseMDM(self.diffusion_cfg, self.gpt, self.processed, self.bs, arch_type="combo", modality="face")

            # losses
            if self.train_modality == "bvh":
                loss_pose = self.diffusion_cfg["loss_pose"]
                if not self.use_diffusion: 
                    for loss in loss_pose:
                        if "lambda_l2" in loss:
                            self.lambda_l2_loss = get_loss_fn(self.diffusion_cfg["loss_factory"][loss])
                else: pass                                                      # Loss computations in gaussian spaced diffusion         
            elif self.train_modality == "face":
                loss_face = self.diffusion_cfg["loss_face"]
                for loss in loss_face:
                    if "mse" in loss:
                        self.ff_mse_loss = get_loss_fn(self.diffusion_cfg["loss_factory"][loss])
            else:
                assert self.train_modality == "both", "Train modality should be either bvh, face or both"
                losses = self.diffusion_cfg["loss"]
                for loss in losses:
                    if "pos_rec_loss" in loss:
                        self.pos_rec_loss = get_loss_fn(self.diffusion_cfg["loss_factory"][loss])
                    if "fac_rec_loss" in loss:
                        self.fac_rec_loss = get_loss_fn(self.diffusion_cfg["loss_factory"][loss])           
            
        elif self.diffusion_type == "diff_latent":
            raise NotImplementedError(f"Diffusion type {self.diffusion_type} not supported.")
                
        else: raise TypeError(f"Diffusion type {self.diffusion_type} not supported.")
        
    def forward(self, batch, mode):
        
        recons_count = 0
        recons, losses = {}, {}
        # # Posenet
        if self.train_modality in ["bvh", "both"]:
            """
            For diffusion forward pose diffusion function for disentangled or combined representation

            :param batch: dict_keys(['pose_content', 'pose_emotion', 'pose_combo', 'wav_mfcc', 
                                     'wav_mfcc_content', 'wav_mfcc_emotion', 'attr', 
                                     'corpus_gpt', 'emo_label', 'old_ts', 'ts', 'base_diff'])
            :param mode: train/ val
            :return: dict_keys(['recons', 'losses'])                            'losses' are None
            """
            if self.bvh_div:
                recons["pose_content"] = self.pos_con(batch, mode)
                recons["pose_emotion"] = self.pos_emo(batch, mode)
                recons_count += 2
            else:
                recons["pose_combo"] = self.pos_comb(batch, mode) 
                recons_count += 1
        # # FaceNet
        if self.train_modality in ["face", "both"]:
            if self.fac_div:
                recons["face_content"] = self.fac_con(batch, mode)
                recons["face_emotion"] = self.fac_emo(batch, mode)
                recons_count += 2
            else:
                recons["face_combo"] = self.fac_comb(batch, mode) 
                recons_count += 1
        # losses
        if self.train_modality == "bvh":                                        # MDM based                        
            if self.bvh_div:
                if not self.use_diffusion:
                    truncated_batch = {k: v for k, v in batch.items() if k in ["pose_content", "pose_emotion"]}
                    loss, rec_wt = self.lambda_l2_loss(recons, truncated_batch)
                    for k, v in loss.items():
                        if k == "pose_content": losses["pose_content"] = v.mean() * rec_wt
                        elif k == "pose_emotion": losses["pose_emotion"] = v.mean() * rec_wt
                        else: raise TypeError(f"Loss type {k} not supported.")
                else: losses["pose_content"], losses["pose_emotion"] = None, None               # None, computations in gaussian diffusion
            else:
                if not self.use_diffusion:
                    truncated_batch = {k: v for k, v in batch.items() if k in ["pose_combo"]}
                    loss, rec_wt = self.lambda_l2_loss(recons, truncated_batch)
                    for k, v in loss.items():
                        if k == "pose_combo": losses["pose_combo"] = v.mean() * rec_wt
                        else: raise TypeError(f"Loss type {k} not supported.")
                else: losses["pose_combo"] = None                                               # None, computations in gaussian diffusion               
        elif self.train_modality == "face":                                     # FF based
            if self.fac_div:
                loss, rec_wt = self.ff_mse_loss(recons["face_content"], batch["face_content"])
                losses["face_content"] = loss * rec_wt
                loss, rec_wt = self.ff_mse_loss(recons["face_emotion"], batch["face_emotion"])
                losses["face_emotion"] = loss * rec_wt
            else:
                loss, rec_wt = self.ff_mse_loss(recons["face_combo"], batch["face_combo"])
                losses["face_combo"] = loss * rec_wt
        else:                                                                   # CaMN based                                       
            assert self.train_modality == "both", "Train modality should be either bvh, face or both"
            if self.bvh_div:
                loss, rec_wt = self.pos_rec_loss(recons["pose_content"], batch["pose_content"])
                losses["pose_content"] = loss * rec_wt
                loss, rec_wt = self.pos_rec_loss(recons["pose_emotion"], batch["pose_emotion"])
                losses["pose_emotion"] = loss * rec_wt
            else:
                loss, rec_wt = self.pos_rec_loss(recons["pose_combo"], batch["pose_combo"])
                losses["pose_combo"] = loss * rec_wt
            if self.fac_div:
                loss, rec_wt = self.fac_rec_loss(recons["face_content"], batch["face_content"])
                losses["face_content"] = loss * rec_wt
                loss, rec_wt = self.fac_rec_loss(recons["face_emotion"], batch["face_emotion"])
                losses["face_emotion"] = loss * rec_wt
            else:
                loss, rec_wt = self.fac_rec_loss(recons["face_combo"], batch["face_combo"])
                losses["face_combo"] = loss * rec_wt
                
        # assert number of entries
        assert len(recons) == len(losses), f"[DIFF] Number of reconstructions and losses do not match: {len(recons)} != {len(losses)}"
        assert len(recons) == recons_count, \
            f"[DIFF] Recons generated/counted: {len(recons)}/{recons_count} are not compliant with div flags: bvh_div={self.bvh_div}, fac_div={self.fac_div}; train modalities={self.train_modality}"
        
        return {
            "recons": recons,
            "losses": losses
        }
    