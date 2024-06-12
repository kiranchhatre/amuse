
import re
import csv
import time
import json
import wandb
import torch
import string
import random
import shutil
import functools
import subprocess
from bvh import Bvh
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from PIL import Image
from smplx import SMPLX
from pytube import YouTube
from torch.nn import init
from pathlib import Path
from pprint import pprint
from decimal import Decimal
from datetime import datetime
from pydub import AudioSegment
from torch.autograd import Variable
from einops import rearrange, repeat
from pytorch3d import transforms as p3d_tfs
from moviepy.editor import VideoFileClip, concatenate_videoclips

from dm.utils.bvh_utils import *
from dm.utils.wav_utils import *
from dm.utils.ldm_evals import train_takes_dict, moglow_order, val_takes_dict, combined_takes_dict
from dm.utils.ldm_evals import all_actors as ldm_eval_all_actors
from models.diffusion.viz.visualizer import Visualizer
from models.diffusion.utils.mdm_fp16_util import MixedPrecisionTrainer
import models.diffusion.utils.mdm_gaussian_diffusion as gd
from models.latent_diffusion.infer_ldm import mapinfo2takes
from models.diffusion.diffusion_eval import (
    EvaluatorWrapper,
    DiffusionEval,
    evaluation
)
from models.diffusion.utils.mdm_resample import create_named_schedule_sampler, LossAwareSampler
from models.diffusion.utils.mdm_respace import SpacedDiffusion, space_timesteps
from models.latent_diffusion.utils.latent_losses import LatentPriorLosses

class trainer():
    
    def __init__(self, config, device, train_loader, val_loader=None, model_path=None, tag=None, logger_cfg=None, model=None, 
                 b_path=None, processed=None, debug=False, EXEC_ON_CLUSTER=False, pretrained_infer=False, sweep=None, metricsmodel=None):
        
        self.config = config
        self.device = device
        self.model = model
        self.parallelism = self.config["TRAIN_PARAM"]["parallelism"]
        self.sweep = sweep
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tag = tag
        self.debug = debug
        self.metricsmodel = metricsmodel
        self.train_modality = self.config["TRAIN_PARAM"][self.tag]["con_emo_div"]["train_modality"] if self.tag == "diffusion" else None
        if self.tag in ["LPDM", "LPDM_infer", "latent_diffusion"]:
            shuffle_type = self.config["TRAIN_PARAM"]["latent_diffusion"]["shuffle_type"]
        else: shuffle_type = self.config["TRAIN_PARAM"][self.tag]["shuffle_type"]
        if self.tag in ["latent_diffusion"]:
            self.audio_con_ae_training = self.config["TRAIN_PARAM"]["audio_con_ae"]["is_training"]
            self.audio_emo_ae_training = self.config["TRAIN_PARAM"]["audio_emo_ae"]["is_training"]
            self.base_audio_ae_training = self.config["TRAIN_PARAM"]["base_audio_ae"]["is_training"]
            assert (self.audio_con_ae_training, self.audio_emo_ae_training, self.base_audio_ae_training) in [(True, False, False), (False, True, False), (False, False, True), (False, False, False)], \
                   "[Trainer LATDIFF init] audio ae training flag mismatch"
        else: self.audio_con_ae_training, self.audio_emo_ae_training, self.base_audio_ae_training = False, False, False
        self.model_dir_name = self.tag + "_" + datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + shuffle_type
        self.model_dir_name += f"_{self.train_modality}-MODAL" if self.train_modality else "" 
        self.logger_cfg = logger_cfg
        self.b_path = b_path                                                    
        self.EXEC_ON_CLUSTER = EXEC_ON_CLUSTER                                  
        self.pretrained_infer = pretrained_infer                                
        self.processed = processed                                              
        self.model_path_r = model_path 
        self.model_path = self.model_path_r / self.model_dir_name
        
        if self.tag in ["latent_diffusion", "LPDM", "LPDM_infer"]:
            self.smplx_data = self.config["TRAIN_PARAM"]["latent_diffusion"]["smplx_data"]
            self.skip_trans = self.config["TRAIN_PARAM"]["latent_diffusion"]["skip_trans"]
            self.train_upper_body = self.config["TRAIN_PARAM"]["latent_diffusion"]["train_upper_body"]
            if self.smplx_data: 
                if self.skip_trans: self.model_path = self.model_path_r / f"{self.model_dir_name}_smplx_skiptrans"
                elif self.train_upper_body: self.model_path = self.model_path_r / f"{self.model_dir_name}_smplx_upperbody"
                else: self.model_path = self.model_path_r / f"{self.model_dir_name}_smplx"
                self.smplx_rep = self.config["TRAIN_PARAM"]["latent_diffusion"]["smplx_rep"]
                self.vtex_displacement = self.config["TRAIN_PARAM"]["latent_diffusion"]["vtex_displacement"]
                if self.vtex_displacement:
                    smpl_paths = self.processed.parents[1] / "body_models/codebase/models/smplx"
                    bs = self.config["TRAIN_PARAM"]["latent_diffusion"]["batch_size"] * self.config["DATA_PARAM"]["Bvh"]["train_pose_framelen"]
                    base_args = dict(num_betas=300, use_pca=False, flat_hand_mean=True, batch_size=1)
                    smpl_male = SMPLX(model_path=str(smpl_paths/"SMPLX_MALE.npz"), **base_args)
                    smpl_female = SMPLX(model_path=str(smpl_paths/"SMPLX_FEMALE.npz"), **base_args)
                    smpl_neutral = SMPLX(model_path=str(smpl_paths/"SMPLX_NEUTRAL.npz"), **base_args)
                    smplx_flame = SMPLX(model_path=str(smpl_paths/"SMPLX_NEUTRAL_2020.npz"), **base_args) 
                    smpl_male, smpl_female, smpl_neutral, smplx_flame = smpl_male.eval(), smpl_female.eval(), smpl_neutral.eval(), smplx_flame.eval()
                    smplx_models = {"male": smpl_male, "female": smpl_female, "neutral": smpl_neutral, "flame": smplx_flame}
                else: smplx_models = None
            else: self.model_path = self.model_path_r / f"{self.model_dir_name}_pymo"
            if self.pretrained_infer:
                self.stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                ST_overwrite = self.config["TRAIN_PARAM"]["test"]["style_transfer"]["overwrite"] if self.config["TRAIN_PARAM"]["test"]["style_transfer"]["use"] else None
                EC_overwrite = self.config["TRAIN_PARAM"]["test"]["emotion_control"]["overwrite"] if self.config["TRAIN_PARAM"]["test"]["emotion_control"]["use"] else None
                SXE_overwrite = self.config["TRAIN_PARAM"]["test"]["style_Xemo_transfer"]["overwrite"] if self.config["TRAIN_PARAM"]["test"]["style_Xemo_transfer"]["use"] else None
                if any([ST_overwrite, EC_overwrite, SXE_overwrite]):
                    overwrite_model_dir_name = ST_overwrite if ST_overwrite else EC_overwrite if EC_overwrite else SXE_overwrite
                    self.stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                    print(f"[Trainer LATDIFF EVAL DIR] overwrite model dir name: {overwrite_model_dir_name} at {self.stamp}")
                    self.model_path = self.model_path_r / overwrite_model_dir_name
        
        if not self.debug:
            Path(self.model_path).mkdir(parents=True, exist_ok=True)
            self._dump_args()
            
        # Task specifics: 
        if self.tag == "wav_dtw_mfcc":
            
            self.fbank_noise = self.config["TRAIN_PARAM"][self.tag]["noise"]
            trainables = [p for p in self.model.parameters() if p.requires_grad]
            self.opt = torch.optim.Adam(trainables, lr=self.config["TRAIN_PARAM"][self.tag]["lr"],
                                        weight_decay=self.config["TRAIN_PARAM"][self.tag]["weight_decay"],
                                        betas=(self.config["TRAIN_PARAM"][self.tag]["beta1"], self.config["TRAIN_PARAM"][self.tag]["beta2"]))
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt, 
                                                                  list(range(self.config["TRAIN_PARAM"][self.tag]["lrscheduler_start"], 1000, self.config["TRAIN_PARAM"][self.tag]["lrscheduler_step"])),
                                                                  gamma=self.config["TRAIN_PARAM"][self.tag]["lrscheduler_gamma"])
            self.ast_ablation = self.config["TRAIN_PARAM"][self.tag]["ablation"]
            self.frame_based_feats = self.config["TRAIN_PARAM"][self.tag]["frame_based_feats"]
        
        elif self.tag in ["LPDM", "LPDM_infer"]:
            # Common
            self.seq_len = self.config["DATA_PARAM"]["Bvh"]["train_pose_framelen"]
            self.fps = self.config["DATA_PARAM"]["Bvh"]["fps"]
            self.model_save_freq = self.config["TRAIN_PARAM"]["latent_diffusion"]["model_save_freq"]
            self.epochs = self.config["TRAIN_PARAM"]["latent_diffusion"]["n_epochs"]
            if self.smplx_data: assert "_smplx_" in self.config["TRAIN_PARAM"]["diffusion"]["lmdb_cache"], "[Trainer LPDM init] smpl data flag and lmdb cache mismatch"
            # Prior
            assert self.config["TRAIN_PARAM"]["motionprior"]["emotional"], "[Trainer LPDM init] Only emotional motion prior is supported"
            if not self.smplx_data: raise ValueError("Pymo based implementation removed.")
            if self.pretrained_infer:
                # Blender
                self.viz_type = self.config["TRAIN_PARAM"]["latent_diffusion"]["viz_type"]
                self.viz_freq = self.config["TRAIN_PARAM"]["latent_diffusion"]["viz_freq"]
                base_visualizer = Visualizer(self.config, self.b_path, self.processed) 
                self.visualizer = base_visualizer.get_visualizer()
                self.visualizer.load_in_blender(self.EXEC_ON_CLUSTER)       
                # Applications
                self.style_transfer = self.config["TRAIN_PARAM"]["test"]["style_transfer"]["use"]
                if self.style_transfer:
                    self.style_transfer_actors = self.config["TRAIN_PARAM"]["test"]["style_transfer"]["actors"]
                    self.style_transfer_emotion = self.config["TRAIN_PARAM"]["test"]["style_transfer"]["emotion"]
                self.emotion_control = self.config["TRAIN_PARAM"]["test"]["emotion_control"]["use"]
                if self.emotion_control:
                    self.emotion_control_actor = self.config["TRAIN_PARAM"]["test"]["emotion_control"]["actor"]
                    self.emotion_control_content_emotion = self.config["TRAIN_PARAM"]["test"]["emotion_control"]["content_emotion"]
                    self.emotion_control_take_element = self.config["TRAIN_PARAM"]["test"]["emotion_control"]["take_element"]
                self.content_control = self.config["TRAIN_PARAM"]["test"]["content_control"]["use"]
                if self.content_control:
                    raise Exception("Content control not supported yet")
                self.style_Xemo_transfer = self.config["TRAIN_PARAM"]["test"]["style_Xemo_transfer"]["use"]
                if self.style_Xemo_transfer: 
                    self.style_Xemo_transfer_actors = self.config["TRAIN_PARAM"]["test"]["style_Xemo_transfer"]["actors"]
                    self.style_Xemo_transfer_emotion = self.config["TRAIN_PARAM"]["test"]["style_Xemo_transfer"]["emotion"]
            else:
                # Latent Diffusion
                cfg_name = self.config["TRAIN_PARAM"]["latent_diffusion"]["arch"]
                with open(str(Path(self.processed.parents[1], f"configs/{cfg_name}.json")), "r") as f: self.lpdm_cfg = json.load(f)
                if self.smplx_data: 
                    self.lpdm_cfg["losses"]["use_recons_joints"] = False
                    self.lpdm_cfg["losses"]["vtex_displacement"] = self.vtex_displacement
                self.ablation_diff_only = self.config["TRAIN_PARAM"]["latent_diffusion"]["ablation_diffusion_only"]
                self.lpdm_losses = LatentPriorLosses(self.lpdm_cfg, self.device, smplx_models)
                optimizer = self.config["TRAIN_PARAM"]["latent_diffusion"]["optimizer_name"]
                lr_base = self.config["TRAIN_PARAM"]["latent_diffusion"]["lr_base"]
                assert optimizer in ["adamw"], "[Trainer LPDM init] Only adamw optimizers are supported for latent diffusion"
                model_params = list(self.model["prior"].parameters()) + list(self.model["ldm"].parameters()) 
                self.lpdm_opt = torch.optim.AdamW(lr=lr_base, params=model_params)
        
        else: raise ValueError(f"Tag {self.tag} not supported")

    def train_dtw_ast(self):
        
        print("INIT: AST-BASED SPEECH CONTENT-EMOTION-STYLE DISENTANGLEMENT TRAINING. TYPE: %s" % self.ast_ablation)
        
        tag_type = self.config["TRAIN_PARAM"]["log_tag"]
        if not self.debug:
            wandb.login(key=self.logger_cfg["WANDB_PARAM"]["api"])
            wandb.init(project=self.logger_cfg["WANDB_PARAM"]["project"], 
                    entity=self.logger_cfg["WANDB_PARAM"]["entity"],
                    tags=self.logger_cfg[self.tag][tag_type], name=self.model_dir_name)
            wandb.config.update(self.config["TRAIN_PARAM"][self.tag])
            wandb.watch(self.model, log="all")
            tags = self.logger_cfg[self.tag][tag_type]
            print(f"[TRAIN W&B] Tags set to: {tags}")

        if self.parallelism == "dp": 
            print(f"[AST T] Using DataParallel with {torch.cuda.device_count()} GPUs")
            scaler = torch.cuda.amp.GradScaler()
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        iter_start_time = time.time()
        # torch.cuda.reset_max_memory_allocated()
        
        for epoch in range(0, self.config["TRAIN_PARAM"][self.tag]["n_epochs"]):
            
            self.model.train()
            train_stats = []
            torch.set_grad_enabled(True)
            train_epoch_loss, train_emo_acc, train_person_id_acc = 0.0, 0.0, 0.0
            val_epoch_loss, val_emo_acc, val_person_id_acc = 0.0, 0.0, 0.0
            train_broad_losses, val_broad_losses = {}, {}
            stat_keys = ["acc", "average_precisions", "f1", "recall"]
            
            # # Debugging
            # samples = 20
            # from torch.utils.data import Subset
            # self.train_loader = torch.utils.data.DataLoader(Subset(self.train_loader.dataset, range(samples)), batch_size=self.config["TRAIN_PARAM"][self.tag]["batch_size"], shuffle=True)
            # self.val_loader = torch.utils.data.DataLoader(Subset(self.val_loader.dataset, range(samples)), batch_size=self.config["TRAIN_PARAM"][self.tag]["batch_size"], shuffle=True)
            
            for i, data in tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Epoch: " + str(epoch+1), leave=False):
            
                if self.parallelism == "none": 
                    for k, v in data.items(): data[k] = v.to(self.device)
                    self.opt.zero_grad()
                    loss_dict = self.model(data, noise=self.fbank_noise, ablation=self.ast_ablation, frame_based_feats=self.frame_based_feats)
                    loss, emo_acc, person_id_acc, broad_losses = loss_dict["loss"], loss_dict["emo_acc"], loss_dict["person_id_acc"], loss_dict["loss_dict"]
                    self.model.zero_grad(set_to_none=True)
                    loss.backward()
                    self.opt.step()
                
                elif self.parallelism == "dp":
                    for k, v in data.items(): data[k] = v.to(self.device, non_blocking=True)
                    self.opt.zero_grad()
                    with torch.cuda.amp.autocast(dtype=torch.float16): loss_dict = self.model(data=data, noise=self.fbank_noise, ablation=self.ast_ablation, frame_based_feats=self.frame_based_feats)
                    loss, emo_acc, person_id_acc, broad_losses = loss_dict["loss"].sum(), loss_dict["emo_acc"].sum(), loss_dict["person_id_acc"].sum(), loss_dict["loss_dict"]
                    for k in broad_losses.keys(): broad_losses[k] = broad_losses[k].sum() # accumulate broad losses for logging
                    self.model.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.step(self.opt)
                    scaler.update()
                
                train_epoch_loss += loss.item() 
                train_emo_acc += emo_acc.item() if not isinstance(emo_acc, float) else emo_acc
                train_person_id_acc += person_id_acc.item() if not isinstance(person_id_acc, float) else person_id_acc
                train_stats.append(loss_dict)
                train_broad_losses = {k: v + train_broad_losses[k] if k in train_broad_losses.keys() else v for k, v in broad_losses.items()}

            train_epoch_loss /= len(self.train_loader)
            train_emo_acc /= len(self.train_loader)
            train_person_id_acc /= len(self.train_loader)
            train_broad_losses = {k: v/len(self.train_loader) for k, v in train_broad_losses.items()}
            
            if isinstance(self.model, nn.DataParallel): stats_results = self.model.module.calculate_stats(train_stats, self.parallelism, ablation=self.ast_ablation)
            else: stats_results = self.model.calculate_stats(train_stats, self.parallelism, ablation=self.ast_ablation) 
            emo_stats, subject_stats = stats_results["emo_stats"], stats_results["subject_stats"]
            print("< ========================= >")
            print(f"[AST-T] Epoch: [{epoch+1}/{self.config['TRAIN_PARAM'][self.tag]['n_epochs']}] time: {(time.time() - iter_start_time)/60.0:.4f} mins")
            print(f"[AST-T] Total Loss: {train_epoch_loss:.8f}")
            print(f"[AST-T] Emotion Acc: {train_emo_acc:.8f}")
            print(f"[AST-T] Person ID Acc: {train_person_id_acc:.8f}")
            for key in train_broad_losses.keys(): print(f"[AST-T] Broad Loss {key}: {train_broad_losses[key]}")
            if emo_stats is not None:
                for key in stat_keys: print(f"[AST-T] Stat Emotion {key}: {emo_stats[key]}")
            if subject_stats is not None:
                for key in stat_keys: print(f"[AST-T] Stat Person ID {key}: {subject_stats[key]}") 

            if not self.debug: 
                wandb.log({"train_loss": train_epoch_loss, "train_emo_acc": train_emo_acc, "train_person_id_acc": train_person_id_acc, "epoch": epoch})
                if emo_stats is not None:
                    for k in stat_keys: wandb.log({f"train_stat_emo_{k}": emo_stats[k], "epoch": epoch})
                if subject_stats is not None:
                    for k in stat_keys: wandb.log({f"train_stat_person_id_{k}": subject_stats[k], "epoch": epoch})
                for k in train_broad_losses.keys(): wandb.log({f"train_broad_loss_{k}": train_broad_losses[k], "epoch": epoch})
               
            self.model.eval() 
            val_stats = []
            with torch.no_grad():
                for i, data in tqdm(enumerate(self.val_loader), total=len(self.val_loader), desc="Epoch: " + str(epoch+1), leave=False):
                    
                    for k, v in data.items(): data[k] = v.to(self.device)
                    loss_dict = self.model(data, noise=self.fbank_noise, ablation=self.ast_ablation, frame_based_feats=self.frame_based_feats)
                    loss, emo_acc, person_id_acc, broad_losses = loss_dict["loss"], loss_dict["emo_acc"], loss_dict["person_id_acc"], loss_dict["loss_dict"]
                    val_epoch_loss += torch.sum(loss).item() if loss.dim() > 0 else loss.item()
                    if isinstance(emo_acc, float): val_emo_acc += emo_acc
                    elif emo_acc.dim() > 0: val_emo_acc += torch.sum(emo_acc).item()
                    else: val_emo_acc += emo_acc.item()
                    if isinstance(person_id_acc, float): val_person_id_acc += person_id_acc
                    elif person_id_acc.dim() > 0: val_person_id_acc += torch.sum(person_id_acc).item()
                    else: val_person_id_acc += person_id_acc.item()
                    if self.parallelism == "dp": 
                        for k in broad_losses.keys(): broad_losses[k] = broad_losses[k].sum()
                    val_stats.append(loss_dict)
                    val_broad_losses = {k: v + val_broad_losses[k] if k in val_broad_losses.keys() else v for k, v in broad_losses.items()}
                
                val_epoch_loss /= len(self.val_loader)
                val_emo_acc /= len(self.val_loader)
                val_person_id_acc /= len(self.val_loader)
                val_broad_losses = {k: v/len(self.val_loader) for k, v in val_broad_losses.items()}
                
                if isinstance(self.model, nn.DataParallel): stats_results = self.model.module.calculate_stats(val_stats, self.parallelism, ablation=self.ast_ablation)
                else: stats_results = self.model.calculate_stats(val_stats, self.parallelism, ablation=self.ast_ablation) 
                emo_stats, subject_stats = stats_results["emo_stats"], stats_results["subject_stats"]
                print(f"[AST-V] Epoch: [{epoch+1}/{self.config['TRAIN_PARAM'][self.tag]['n_epochs']}] time: {(time.time() - iter_start_time)/60.0:.4f} mins")
                print(f"[AST-V] Total Loss: {val_epoch_loss:.8f}")
                print(f"[AST-V] Emotion Acc: {val_emo_acc:.8f}")
                print(f"[AST-V] Person ID Acc: {val_person_id_acc:.8f}")
                for key in val_broad_losses.keys(): print(f"[AST-V] Broad Loss {key}: {val_broad_losses[key]}")
                if emo_stats is not None:
                    for key in stat_keys: print(f"[AST-V] Stat Emotion {key}: {emo_stats[key]}")
                if subject_stats is not None:
                    for key in stat_keys: print(f"[AST-V] Stat Person ID {key}: {subject_stats[key]}")
                
                if not self.debug: 
                    wandb.log({"val_loss": val_epoch_loss, "val_emo_acc": val_emo_acc, "val_person_id_acc": val_person_id_acc, "epoch": epoch})
                    if emo_stats is not None:
                        for k in stat_keys: wandb.log({f"val_stat_emo_{k}": emo_stats[k], "epoch": epoch})
                    if subject_stats is not None:
                        for k in stat_keys: wandb.log({f"val_stat_person_id_{k}": subject_stats[k], "epoch": epoch})
                    for k in val_broad_losses.keys(): wandb.log({f"val_broad_loss_{k}": val_broad_losses[k], "epoch": epoch})
                    
            if not self.debug: torch.save(self.model.state_dict(), str(self.model_path) + "/model_%d_tL%.8f_tEA%.8f_tPA%.8f_vL%.8f_vEA%.8f_vPA%.8f.pkl" % (epoch, train_epoch_loss, train_emo_acc, train_person_id_acc, val_epoch_loss, val_emo_acc, val_person_id_acc))

            self.scheduler.step()
            print(f"[AST] Epoch: [{epoch+1}/{self.config['TRAIN_PARAM'][self.tag]['n_epochs']}] time: {(time.time() - iter_start_time)/60.0:.4f} mins, lr: {self.scheduler.get_last_lr()[0]:.8f}")
            
        print("[AST] Training finished, total time: %4.4f mins" % ((time.time() - iter_start_time)/60.0))

    def train_prior_latdiff_forward_backward_v2(self, baseline=False, lmdb_id=None, verbose=False, audio_ablation=None):
        
        if baseline: print("INIT: BASELINE PRIOR LATENT DIFFUSION TRAINING")
        else: print("INIT: PRIOR LATENT DIFFUSION TRAINING")
        
        if not self.debug:
            tag_type = self.config["TRAIN_PARAM"]["log_tag"]                  
            wandb.login(key=self.logger_cfg["WANDB_PARAM"]["api"])
            wandb.init(project=self.logger_cfg["WANDB_PARAM"]["project"], 
                    entity=self.logger_cfg["WANDB_PARAM"]["entity"],
                    tags=self.logger_cfg[self.tag][tag_type], name=self.model_dir_name)
            wandb.config.update(self.cfg_dump)    
            wandb.watch(self.model["prior"], log="all")
            wandb.watch(self.model["ldm"], log="all")
            tags = self.logger_cfg[self.tag][tag_type]
            print(f"[TRAIN W&B] Tags set to: {tags}")
        
        iter_start_time = time.time()
        for epoch in range(self.epochs):
            
            for _, batch in enumerate(tqdm(self.train_loader, desc="[LPDM-T]", leave=False)):
                
                self.model["prior"].train()
                self.model["ldm"].train()
                torch.set_grad_enabled(True)
                
                for k, v in batch.items(): batch[k] = v.to(self.device) if k not in ["ld_attr"] else v
                
                if self.smplx_rep == "6D":
                    poses, trans = batch["ld_motion"][:, :, :-3], batch["ld_motion"][:, :, -3:]
                    poses = rearrange(poses, "b s (j c) -> b s j c", j=55, c=3)
                    mat = p3d_tfs.axis_angle_to_matrix(poses)   
                    rot6D = p3d_tfs.matrix_to_rotation_6d(mat) 
                    rot6D = rearrange(rot6D, "b s j c -> b s (j c)")
                    batch["ld_motion"] = torch.cat((rot6D, trans), dim=-1) # bs, seq, 333
                else: assert self.smplx_rep == "3D", f"Invalid smplx representation: {self.smplx_rep}"
                
                # prior
                motion = batch["ld_motion"]
                if self.skip_trans: motion = motion[:, :, :-3]
                elif self.train_upper_body:
                    lower_body_jts = [1, 2, 4, 5, 7, 8, 10, 11]
                    no_root_trans = motion[:, :, :-3]
                    motion = no_root_trans.reshape(no_root_trans.shape[0], no_root_trans.shape[1], -1, 55)
                    motion = motion[:, :, :, [i for i in range(55) if i not in lower_body_jts]]
                    motion = motion.reshape(motion.shape[0], motion.shape[1], -1)
                lengths = [self.seq_len] * motion.shape[0]
                motion_z, dist_m = self.model["prior"].encode(features=motion, lengths=lengths)
                feats_rst = self.model["prior"].decode(z=motion_z, lengths=lengths)
                mu_ref = torch.zeros_like(dist_m.loc)
                scale_ref = torch.ones_like(dist_m.scale)
                dist_ref = torch.distributions.Normal(mu_ref, scale_ref)
                if not self.smplx_data:
                    joints_rst = pymo_feats2joints(feats_rst, self.pymo_preprocessed_prior, self.device)
                    joints_ref = pymo_feats2joints(motion, self.pymo_preprocessed_prior, self.device)
                
                # latent diffusion
                ld_audio_con = batch["ld_audio_con"]
                ld_audio_emo = batch["ld_audio_emo"]
                ld_audio_sty = batch["ld_audio_sty"]
                
                if lmdb_id: # ablation variants
                    kind = lmdb_id.split("/")[-1].split("_")[-3]
                    if kind == "feat": kind = lmdb_id.split("/")[-1].split("_")[-5]
                    if kind in ["emotion", "baseline"]: ld_audio_sty = None
                    elif kind == "identity": ld_audio_emo = None
                    else: assert kind == "full", f"Invalid lmdb_id: {lmdb_id}"
                    if verbose: print(f"LMDB ID: {kind}")
                
                ld_audio = batch["ld_audio"]
                if baseline: raise NotImplementedError("Baseline for Amuse LPDM not implemented!") # ld_audio_mfcc = audio2slicedmfcc(self.config, batch_raw_wf=ld_audio)
                else: ld_audio_mfcc = None
                with torch.no_grad():
                    inferred_motion_z, _ = self.model["prior"].encode(features=motion, lengths=lengths)
                n_set = self.model["ldm"].diffusion_forward(inferred_motion_z, ld_audio_con, ld_audio_emo, ld_audio_sty, 
                                                            lengths=lengths, ld_audio_mfcc=ld_audio_mfcc)
                
                # inverse diffusion
                with torch.no_grad():
                    noise2z = self.model["ldm"].diffusion_backward(ld_audio_con, ld_audio_emo, ld_audio_sty, ld_audio_mfcc, motion.shape[0])
                    noise2feats_rst = self.model["prior"].decode(z=noise2z, lengths=lengths)
                if not self.smplx_data: noise2joints_rst = pymo_feats2joints(noise2feats_rst, self.pymo_preprocessed_prior, self.device)
                
                motion_3D, feats_rst_3D, noise2feats_rst_3D = dict(), dict(), dict()
                if self.smplx_rep == "6D":
                    for F, AA in zip([motion, feats_rst, noise2feats_rst], [motion_3D, feats_rst_3D, noise2feats_rst_3D]):
                        # F = motion A = motion_3D
                        rot_6D, trans = F[:, :, :-3], F[:, :, -3:]
                        rot_6D = rearrange(rot_6D, "b s (j c) -> b s j c", c=6)
                        mat = p3d_tfs.rotation_6d_to_matrix(rot_6D)
                        poses = p3d_tfs.matrix_to_axis_angle(mat)
                        AA["poses"], AA["trans"] = poses, trans
                else:
                    for F, AA in zip([motion, feats_rst, noise2feats_rst], [motion_3D, feats_rst_3D, noise2feats_rst_3D]):
                        AA["poses"], AA["trans"] = F[:, :, :-3], F[:, :, -3:]
                        AA["poses"] = rearrange(AA["poses"], "b s (j c) -> b s j c", c=3)
                
                rs_set = {
                    "m_ref": motion,
                    "m_rst": feats_rst,
                    "dist_m": dist_m,
                    "dist_ref": dist_ref,
                    "joints_ref": joints_ref if not self.smplx_data else None,
                    "joints_rst": joints_rst if not self.smplx_data else None,
                    "noise_pred": n_set["noise_pred"],
                    "noise": n_set["noise"],
                    "gen_m_rst": noise2feats_rst,
                    "gen_joints_rst": noise2joints_rst if not self.smplx_data else None,
                    "attr": batch["ld_attr"],
                    "m_ref_3D": motion_3D,
                    "m_rst_3D": feats_rst_3D,
                    "gen_m_rst_3D": noise2feats_rst_3D
                }
                loss = self.lpdm_losses.update(rs_set, audio_ablation=audio_ablation)
                self.lpdm_opt.zero_grad()
                self.model["prior"].zero_grad(set_to_none=True)
                self.model["ldm"].zero_grad(set_to_none=True)
                loss.backward()
                self.lpdm_opt.step()
            
            loss_dict = self.lpdm_losses.compute()
            self.lpdm_losses.reset()  
            if not self.vtex_displacement:  
                loss_dict['gen_vtex_displacement'] = 0.0
                loss_dict['rec_vtex_displacement'] = 0.0
            print(f"[LPDM-T] Epoch: [{epoch+1}/{self.epochs}] t: {time.time() - iter_start_time:.4f} s, rec_feat: {loss_dict['recons_feature']:.8f}, kl: {loss_dict['kl_motion']:.8f}, rec_jts: {loss_dict['recons_joints']:.8f}, inst_loss: {loss_dict['inst_loss']:.8f}, gen_feature: {loss_dict['gen_feature']:.8f}, gen_joints: {loss_dict['gen_joints']:.8f}, rec_vtex: {loss_dict['rec_vtex_displacement']:.8f}, gen_vtex: {loss_dict['gen_vtex_displacement']:.8f}, total: {loss_dict['total']:.8f}", flush=True)    
            
            if not self.debug:
                for k, v in loss_dict.items(): 
                    v = v.item() if isinstance(v, torch.Tensor) else v
                    wandb.log({"train_" + k: v, "epoch": epoch})
        
            # save model
            if all([not self.debug, (epoch+1) % self.model_save_freq == 0]):
                print(f"[LPDM-T] Saving model at epoch {epoch+1}", flush=True)
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model["prior"].state_dict()
                }, str(self.model_path) + "/prior_model_NoOpt_recF{:.4f}_recJ{:.4f}_kl{:.4f}_genF{:.4f}_genJ{:.4f}_instL{:.4f}_vtexR{:.4f}_vtexG{:.4f}_total{:.4f}_e{}.pt".format(loss_dict["recons_feature"], 
                                                                                                                                                                                loss_dict["recons_joints"], 
                                                                                                                                                                                loss_dict["kl_motion"], 
                                                                                                                                                                                loss_dict["gen_feature"], 
                                                                                                                                                                                loss_dict["gen_joints"], 
                                                                                                                                                                                loss_dict["inst_loss"],
                                                                                                                                                                                loss_dict["rec_vtex_displacement"],
                                                                                                                                                                                loss_dict["gen_vtex_displacement"],
                                                                                                                                                                                loss_dict["total"], 
                                                                                                                                                                                epoch+1))
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model["ldm"].state_dict(),
                    "optimizer_state_dict": self.lpdm_opt.state_dict()
                }, str(self.model_path) + "/latdiff_model_wOpt_recF{:.4f}_recJ{:.4f}_kl{:.4f}_genF{:.4f}_genJ{:.4f}_instL{:.4f}_vtexR{:.4f}_vtexG{:.4f}_total{:.4f}_e{}.pt".format(loss_dict["recons_feature"], 
                                                                                                                                                                                loss_dict["recons_joints"], 
                                                                                                                                                                                loss_dict["kl_motion"], 
                                                                                                                                                                                loss_dict["gen_feature"], 
                                                                                                                                                                                loss_dict["gen_joints"], 
                                                                                                                                                                                loss_dict["inst_loss"],
                                                                                                                                                                                loss_dict["rec_vtex_displacement"],
                                                                                                                                                                                loss_dict["gen_vtex_displacement"],
                                                                                                                                                                                loss_dict["total"], 
                                                                                                                                                                                epoch+1))
                
        print("[LPDM] Training finished, total time elapsed: %4.4f mins" % ((time.time() - iter_start_time)/60.0)) 

    def _infer_prior_latdiff_from_audio_v1(self, baseline, ldm_epoch, audio_list, short_audio_list, modelversion, ammetric):
        
        start_time = time.time()
        task = self.config["TRAIN_PARAM"]["baselines"]["renders"]["task"]
        
        if task == "custom_renders":
            # NOTE! Make sure each audio is a 10 sec wav file
            audios_r = Path(self.config["TRAIN_PARAM"]["baselines"]["renders"]["custom_audios"])
            target_path = Path(self.config["TRAIN_PARAM"]["baselines"]["renders"]["custom_renders"])
            
            for rep_i in range(self.config["TRAIN_PARAM"]["test"]["replication_times"]): # seed change
                print(f" <===== INIT: AUDIO LIST LPDM EVALUATION, REP {rep_i+1}/{self.config['TRAIN_PARAM']['test']['replication_times']} =====>")
            
                if baseline: raise Exception("Baseline not implemented")
                else:
                    audios = audios_r.glob("*.wav")
                    for audio in audios:
                        rst = []
                        actor = "scott"
                        sliced_audio_pydub = AudioSegment.from_wav(str(audio))
                        audio_arr, _ = torchaudio.load(str(audio))
                        audio_arr = audio_arr - audio_arr.mean()
                        z_con, z_emo, z_sty = self.model.process_single_seq(audio_arr, framerate=16000, baseline=baseline)
                        feats_rst = self.model.diffusion_backward(1, z_con, z_emo, z_sty)
                        poses, trans = feats_rst["poses"], feats_rst["trans"]
                        poses = rearrange(poses, "b t j d -> b t (j d)")
                        feats_rst = torch.cat((poses,trans), dim=-1)
                        rst.append(
                        {
                            "feats": feats_rst,
                            "audio": sliced_audio_pydub,
                            "info": actor
                        })
                        
                        video_dump_r = target_path / f"Custom_audios_{self.stamp}_E{ldm_epoch}" / f"rep{rep_i}"
                        assert self.viz_type in ["CaMN"], "[LDM EVAL] Invalid viz type: [%s]" % self.viz_type
                        for i, sample_dict in enumerate(rst):
                            print(f"VISUALIZATION: LIST AUDIOS {i} =====>")
                            video_dump = video_dump_r / f"rst_{i}"
                            self.visualizer.animate_ldm_sample_v1(sample_dict, video_dump, self.smplx_data, self.skip_trans, without_txt=True)
        
        else: raise NotImplementedError(f"Implement your own task: {task}")
        
        print(f"[LDM EVAL] Audio list inference done, total time elapsed: {time.time() - start_time:.4f} s")

    def eval_prior_latdiff_forward_backward_v1(self, baseline, ldm_epoch, audio_list, short_audio_list=False, modelversion=None, ammetric=False):
        
        if audio_list: self._infer_prior_latdiff_from_audio_v1(baseline, ldm_epoch, audio_list, short_audio_list, modelversion, ammetric); return 
        
        metrics_only = self.config["TRAIN_PARAM"]["motion_extractor"]["metrics_only"]
        for rep_i in range(self.config["TRAIN_PARAM"]["test"]["replication_times"]):
            print(f" <===== INIT: LPDM EVALUATION, REP {rep_i+1}/{self.config['TRAIN_PARAM']['test']['replication_times']} =====>")
            # eval_data = self.model.process_loader(self.train_loader)
            eval_data = self.model.process_loader(self.train_loader) # added other model support: evp, emotion only
            metrics = dict()
           
            # if not baseline:
            if modelversion == "full":
                
                if self.style_Xemo_transfer:
                    
                    metrics["style_Xemo_transfer"] = []
                    data = eval_data["style_Xemo_transfer"]
                    actor1, actor2 = self.style_Xemo_transfer_actors[1:-1].split("-")[0], self.style_Xemo_transfer_actors[1:-1].split("-")[1]
                    takes = data["takes"]
                    take1, take2, take3, take4 = takes.split("*")[0], takes.split("*")[1], takes.split("*")[2], takes.split("*")[3]
                    assert take1 == take3 and take2 == take4, f"[LDM EVAL] Takes: {takes} not aligned"
                    
                    attr1 = data[actor1][take1]["ld_attr"]
                    a1_attr = f"{attr1[0]} {attr1[1]} {attr1[2]} {attr1[4]}"
                    attr2 = data[actor2][take2]["ld_attr"]
                    a2_attr = f"{attr2[0]} {attr2[1]} {attr2[2]} {attr2[4]}"
                    
                    eval_set = (
                        (actor1, take1, data[actor1][take1]["ld_z"], data[actor1][take1]["ld_wav"], data[actor1][take1]["ld_z_con"], data[actor1][take1]["ld_z_emo"], data[actor1][take1]["ld_z_sty"], "", data[actor1][take1]["ld_motion"]),
                        (actor2, take1, data[actor2][take3]["ld_z"], data[actor2][take3]["ld_wav"], data[actor2][take3]["ld_z_con"], data[actor2][take3]["ld_z_emo"], data[actor2][take3]["ld_z_sty"], "", data[actor2][take3]["ld_motion"]),
                        (actor1, take1, data[actor1][take1]["ld_z"], data[actor1][take1]["ld_wav"], data[actor1][take1]["ld_z_con"], data[actor1][take1][f"ld_z_emo_{actor2}_{take4}"], data[actor1][take1][f"ld_z_sty_{actor2}_{take4}"], f"{actor1}_{take1}_to_{actor2}_{take4}", data[actor1][take1]["ld_motion"]),
                        (actor2, take1, data[actor2][take3]["ld_z"], data[actor2][take3]["ld_wav"], data[actor2][take3]["ld_z_con"], data[actor2][take3][f"ld_z_emo_{actor1}_{take2}"], data[actor2][take3][f"ld_z_sty_{actor1}_{take2}"], f"{actor2}_{take3}_to_{actor1}_{take2}", data[actor2][take3]["ld_motion"]),
                        
                        (actor1, take2, data[actor1][take2]["ld_z"], data[actor1][take2]["ld_wav"], data[actor1][take2]["ld_z_con"], data[actor1][take2]["ld_z_emo"], data[actor1][take2]["ld_z_sty"], "", data[actor1][take2]["ld_motion"]),
                        (actor2, take2, data[actor2][take4]["ld_z"], data[actor2][take4]["ld_wav"], data[actor2][take4]["ld_z_con"], data[actor2][take4]["ld_z_emo"], data[actor2][take4]["ld_z_sty"], "", data[actor2][take4]["ld_motion"]),
                        (actor1, take2, data[actor1][take2]["ld_z"], data[actor1][take2]["ld_wav"], data[actor1][take2]["ld_z_con"], data[actor1][take2][f"ld_z_emo_{actor2}_{take3}"], data[actor1][take2][f"ld_z_sty_{actor2}_{take3}"], f"{actor1}_{take2}_to_{actor2}_{take3}", data[actor1][take2]["ld_motion"]),
                        (actor2, take2, data[actor2][take4]["ld_z"], data[actor2][take4]["ld_wav"], data[actor2][take4]["ld_z_con"], data[actor2][take4][f"ld_z_emo_{actor1}_{take1}"], data[actor2][take4][f"ld_z_sty_{actor1}_{take1}"], f"{actor2}_{take4}_to_{actor1}_{take1}", data[actor2][take4]["ld_motion"]),
                    )

                    rst = []
                    for _, (actor, take, z, audio, z_con, z_emo, z_sty, swap, src_motion) in enumerate(tqdm(eval_set, desc=f"Eval set style X emo transfer")):
                            
                        act_attr = a1_attr if actor in a1_attr else a2_attr
                        tt = take.split("_")[-1]
                        rst_info = f"Style X Emo Transfer - {act_attr} yrs {tt} {self.style_Xemo_transfer_emotion[1:-1]}"
                        none_None = []
                        if swap:
                            swap_attr = a1_attr if swap in a1_attr else a2_attr
                            swap_info = f"Swapped - {swap_attr} yrs {tt} {self.style_Xemo_transfer_emotion[1:-1]}"
                            for ii in [z_con, z_emo, z_sty]: 
                                if ii is not None: none_None.append(ii)
                            bsz = min([ii.shape[0] for ii in none_None])
                            # bsz = min(z.shape[0], z_con.shape[0], z_emo.shape[0], z_sty.shape[0])
                            z_con = z_con[:bsz]
                            z_emo = z_emo[:bsz] if z_emo is not None else None
                            z_sty = z_sty[:bsz] if z_sty is not None else None
                        else:
                            swap_info = "Not swapped, original"
                            bsz = z.shape[0]
                        feats_rst = self.model.diffusion_backward(bsz, z_con, z_emo, z_sty)
                        sliced_audio = audio[0:bsz*10000]
                        poses, trans = feats_rst["poses"], feats_rst["trans"]
                        poses = rearrange(poses, "b t j d -> b t (j d)")
                        feats_rst = torch.cat((poses,trans), dim=-1)
                        rst.append(
                            {
                            "feats": feats_rst,
                            "audio": sliced_audio,
                            "info": rst_info,
                            "swap_info": swap_info
                            }
                        )
                        metrics["style_Xemo_transfer"].append(
                            {
                                "actor": actor,
                                "take": take,
                                "swap_info": swap_info,
                                "rst_info": rst_info,
                                "src_motion": src_motion,
                                "rst_motion": feats_rst,
                                "audio": sliced_audio,
                                "z_con": z_con,
                                "z_emo": z_emo,
                                "z_sty": z_sty
                            }
                        )
                        
                    if not metrics_only:        
                        run_info = f"{self.style_Xemo_transfer_actors[1:-1]}_{self.style_Xemo_transfer_emotion[1:-1]}"
                        video_dump_r = self.model_path / "viz" / f"style_Xemo_transfer_{run_info}_{self.stamp}_E{ldm_epoch}" / f"rep{rep_i}"
                        assert self.viz_type in ["CaMN"], "[LDM EVAL] Invalid viz type: [%s]" % self.viz_type
                        for i, sample_dict in enumerate(rst):
                            print(f"VISUALIZATION: STYLE TRANSFER {i} =====>")
                            video_dump = video_dump_r / f"rst_{i}"
                            self.visualizer.animate_ldm_sample_v2(sample_dict, video_dump)
                        
                        
                        # if not self.EXEC_ON_CLUSTER:
                        if True: # use custom_vid_concats.py if errors on cluster side
                        
                            print(f"VISUALIZATION: STYLE TRANSFER COMBINED =====>")
                            
                            rst_dirs = [f for f in video_dump_r.iterdir() if f.is_dir()]
                            rst_dirs = sorted(rst_dirs, key=lambda x: int(x.name.split("_")[-1]))
                            take_1_gen, take_2_gen = rst_dirs[0:4], rst_dirs[4:8]
                            min_t1_mp4 = min([len(list(gen.glob("*/*.mp4"))) for gen in take_1_gen]) 
                            min_t2_mp4 = min([len(list(gen.glob("*/*.mp4"))) for gen in take_2_gen]) 
                            
                            combined_video_path = video_dump_r / "combined"
                            combined_video_path.mkdir(parents=True, exist_ok=True)
                            
                            for i in range(min_t1_mp4):
                                # TODO: do not generate seq 0 and -1, during pauses and no driving speech
                                combined_video_file = combined_video_path / f"combined_0_set_{i}.mp4"
                                
                                vid1 = next(Path(take_1_gen[0], f"seq_{i}").glob("*.mp4"))         
                                vid2 = next(Path(take_1_gen[1], f"seq_{i}").glob("*.mp4"))
                                vid3 = next(Path(take_1_gen[2], f"seq_{i}").glob("*.mp4"))
                                vid4 = next(Path(take_1_gen[3], f"seq_{i}").glob("*.mp4"))
                                
                                _ = subprocess.call([
                                    "ffmpeg", "-i", str(vid1), "-i", str(vid2), "-i", str(vid3), "-i", str(vid4), "-filter_complex", "[0:v][1:v]hstack[top];[2:v][3:v]hstack[bottom];[top][bottom]vstack,format=yuv420p[v]", "-map", "[v]", str(combined_video_file)
                                ])
                                combined_video_file_Audio = combined_video_path / f"combined_0_set_{i}_audio_1.mp4"
                                _ = subprocess.call([
                                    "ffmpeg", "-i", str(combined_video_file), "-i", str(vid1), "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0", "-shortest", str(combined_video_file_Audio)
                                ])
                                combined_video_file_Audio = combined_video_path / f"combined_0_set_{i}_audio_2.mp4"
                                _ = subprocess.call([
                                    "ffmpeg", "-i", str(combined_video_file), "-i", str(vid2), "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0", "-shortest", str(combined_video_file_Audio)
                                ])
                                combined_video_file.unlink()
                            
                            for i in range(min_t2_mp4):
                                
                                combined_video_file = combined_video_path / f"combined_1_set_{i}.mp4"
                                
                                vid1 = next(Path(take_2_gen[0], f"seq_{i}").glob("*.mp4"))         
                                vid2 = next(Path(take_2_gen[1], f"seq_{i}").glob("*.mp4"))
                                vid3 = next(Path(take_2_gen[2], f"seq_{i}").glob("*.mp4"))
                                vid4 = next(Path(take_2_gen[3], f"seq_{i}").glob("*.mp4"))
                                
                                _ = subprocess.call([
                                    "ffmpeg", "-i", str(vid1), "-i", str(vid2), "-i", str(vid3), "-i", str(vid4), "-filter_complex", "[0:v][1:v]hstack[top];[2:v][3:v]hstack[bottom];[top][bottom]vstack,format=yuv420p[v]", "-map", "[v]", str(combined_video_file)
                                ])
                                combined_video_file_Audio = combined_video_path / f"combined_0_set_{i}_audio_1.mp4"
                                _ = subprocess.call([
                                    "ffmpeg", "-i", str(combined_video_file), "-i", str(vid1), "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0", "-shortest", str(combined_video_file_Audio)
                                ])
                                combined_video_file_Audio = combined_video_path / f"combined_0_set_{i}_audio_2.mp4"
                                _ = subprocess.call([
                                    "ffmpeg", "-i", str(combined_video_file), "-i", str(vid2), "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0", "-shortest", str(combined_video_file_Audio)
                                ])
                                combined_video_file.unlink()
                                
                        
                        print(f"END VISUALIZATION: STYLE X-EMO TRANSFER {rep_i+1}/{self.config['TRAIN_PARAM']['test']['replication_times']} =====>")
                    else: print(f"END EVALUATION METRICS ONLY: STYLE X-EMO TRANSFER {rep_i+1}/{self.config['TRAIN_PARAM']['test']['replication_times']} =====>")
                
                if self.style_transfer:
                    
                    metrics["style_transfer"] = []
                    data = eval_data["style_transfer"]
                    actor1, actor2 = self.style_transfer_actors[1:-1].split("-")[0], self.style_transfer_actors[1:-1].split("-")[1]
                    takes = mapinfo2takes(self.style_transfer_emotion, True)
                    take1, take2 = takes[0], takes[1]
                    
                    attr1 = data[actor1][take1]["ld_attr"]
                    a1_attr = f"{attr1[0]} {attr1[1]} {attr1[2]} {attr1[4]}"
                    attr2 = data[actor2][take2]["ld_attr"]
                    a2_attr = f"{attr2[0]} {attr2[1]} {attr2[2]} {attr2[4]}"
                    
                    eval_set = (
                        (actor1, take1, data[actor1][take1]["ld_z"], data[actor1][take1]["ld_wav"], data[actor1][take1]["ld_z_con"], data[actor1][take1]["ld_z_emo"], data[actor1][take1]["ld_z_sty"], "", data[actor1][take1]["ld_motion"]),
                        (actor2, take1, data[actor2][take1]["ld_z"], data[actor2][take1]["ld_wav"], data[actor2][take1]["ld_z_con"], data[actor2][take1]["ld_z_emo"], data[actor2][take1]["ld_z_sty"], "", data[actor2][take1]["ld_motion"]),
                        (actor1, take1, data[actor1][take1]["ld_z"], data[actor1][take1]["ld_wav"], data[actor1][take1]["ld_z_con"], data[actor1][take1][f"ld_z_emo_{actor2}"], data[actor1][take1][f"ld_z_sty_{actor2}"], f"{actor2}", data[actor1][take1]["ld_motion"]), 
                        (actor2, take1, data[actor2][take1]["ld_z"], data[actor2][take1]["ld_wav"], data[actor2][take1]["ld_z_con"], data[actor2][take1][f"ld_z_emo_{actor1}"], data[actor2][take1][f"ld_z_sty_{actor1}"], f"{actor1}", data[actor2][take1]["ld_motion"]),
                        
                        (actor1, take2, data[actor1][take2]["ld_z"], data[actor1][take2]["ld_wav"], data[actor1][take2]["ld_z_con"], data[actor1][take2]["ld_z_emo"], data[actor1][take2]["ld_z_sty"], "", data[actor1][take2]["ld_motion"]),
                        (actor2, take2, data[actor2][take2]["ld_z"], data[actor2][take2]["ld_wav"], data[actor2][take2]["ld_z_con"], data[actor2][take2]["ld_z_emo"], data[actor2][take2]["ld_z_sty"], "", data[actor2][take2]["ld_motion"]),
                        (actor1, take2, data[actor1][take2]["ld_z"], data[actor1][take2]["ld_wav"], data[actor1][take2]["ld_z_con"], data[actor1][take2][f"ld_z_emo_{actor2}"], data[actor1][take2][f"ld_z_sty_{actor2}"], f"{actor2}", data[actor1][take2]["ld_motion"]),
                        (actor2, take2, data[actor2][take2]["ld_z"], data[actor2][take2]["ld_wav"], data[actor2][take2]["ld_z_con"], data[actor2][take2][f"ld_z_emo_{actor1}"], data[actor2][take2][f"ld_z_sty_{actor1}"], f"{actor1}", data[actor2][take2]["ld_motion"]),
                    )
                    
                    rst = []
                    for _, (actor, take, z, audio, z_con, z_emo, z_sty, swap, src_motion) in enumerate(tqdm(eval_set, desc=f"Eval set style transfer")):
                        
                        act_attr = a1_attr if actor in a1_attr else a2_attr
                        tt = take.split("_")[-1]
                        rst_info = f"Style Transfer - {act_attr} yrs {tt} {self.style_transfer_emotion[1:-1]}"
                        if swap:
                            swap_attr = a1_attr if swap in a1_attr else a2_attr
                            swap_info = f"Swapped - {swap_attr} yrs {tt} {self.style_transfer_emotion[1:-1]}"
                            bsz = min(z.shape[0], z_con.shape[0], z_emo.shape[0], z_sty.shape[0])  # time frames differ for fast and slow speakers, we focus on emotional style transfer latent
                            z_con = z_con[:bsz]
                            z_emo = z_emo[:bsz]
                            z_sty = z_sty[:bsz]
                        else:
                            swap_info = "Not swapped, original"
                            bsz = z.shape[0]
                        feats_rst = self.model.diffusion_backward(bsz, z_con, z_emo, z_sty)
                        sliced_audio = audio[0:bsz*10000]
                        poses, trans = feats_rst["poses"], feats_rst["trans"]
                        poses = rearrange(poses, "b t j d -> b t (j d)")
                        feats_rst = torch.cat((poses,trans), dim=-1)
                        rst.append(
                            {
                                "feats": feats_rst,
                                "audio": sliced_audio,
                                "info": rst_info,
                                "swap_info": swap_info
                            }
                        )
                        metrics["style_transfer"].append(
                            {
                                "actor": actor,
                                "take": take,
                                "swap_info": swap_info,
                                "rst_info": rst_info,
                                "src_motion": src_motion,
                                "rst_motion": feats_rst,
                                "audio": sliced_audio,
                                "z_con": z_con,
                                "z_emo": z_emo,
                                "z_sty": z_sty
                            }
                        )
                    
                    if not metrics_only:
                        run_info = f"{self.style_transfer_actors[1:-1]}_{self.style_transfer_emotion[1:-1]}"
                        video_dump_r = self.model_path / "viz" / f"style_transfer_{run_info}_{self.stamp}_E{ldm_epoch}" / f"rep{rep_i}"
                        assert self.viz_type in ["CaMN"], "[LDM EVAL] Invalid viz type: [%s]" % self.viz_type
                        for i, sample_dict in enumerate(rst):
                            print(f"VISUALIZATION: STYLE TRANSFER {i} =====>")
                            video_dump = video_dump_r / f"rst_{i}"
                            self.visualizer.animate_ldm_sample_v2(sample_dict, video_dump, self.smplx_data, self.skip_trans)
                        
                        if not self.EXEC_ON_CLUSTER: # custom_vid_concats.py if breaks
                        
                            print(f"VISUALIZATION: STYLE TRANSFER COMBINED =====>")
                            
                            rst_dirs = [f for f in video_dump_r.iterdir() if f.is_dir()]
                            rst_dirs = sorted(rst_dirs, key=lambda x: int(x.name.split("_")[-1]))
                            take_1_gen, take_2_gen = rst_dirs[0:4], rst_dirs[4:8]
                            min_t1_mp4 = min([len(list(gen.glob("*/*.mp4"))) for gen in take_1_gen]) 
                            min_t2_mp4 = min([len(list(gen.glob("*/*.mp4"))) for gen in take_2_gen]) 
                            
                            combined_video_path = video_dump_r / "combined"
                            combined_video_path.mkdir(parents=True, exist_ok=True)
                            
                            for i in range(min_t1_mp4):
                                # TODO: do not generate seq 0 and -1, during pauses and no driving speech
                                combined_video_file = combined_video_path / f"combined_0_set_{i}.mp4"
                                
                                vid1 = next(Path(take_1_gen[0], f"seq_{i}").glob("*.mp4"))         
                                vid2 = next(Path(take_1_gen[1], f"seq_{i}").glob("*.mp4"))
                                vid3 = next(Path(take_1_gen[2], f"seq_{i}").glob("*.mp4"))
                                vid4 = next(Path(take_1_gen[3], f"seq_{i}").glob("*.mp4"))
                                
                                for v in [vid1, vid2, vid3, vid4]:
                                    _ = subprocess.call([
                                        "ffmpeg", "-i", str(v), "-c", "copy", "-bsf:v", "h264_mp4toannexb", "-f", "mpegts", str(v.with_suffix(".ts"))
                                    ])

                                _ = subprocess.call([
                                    "ffmpeg", "-i", f"concat:{str(vid1.with_suffix('.ts'))}|{str(vid2.with_suffix('.ts'))}|{str(vid3.with_suffix('.ts'))}|{str(vid4.with_suffix('.ts'))}", "-c", "copy", "-bsf:a", "aac_adtstoasc", str(combined_video_file)
                                ])
                                
                                for v in [vid1, vid2, vid3, vid4]: v.with_suffix(".ts").unlink()
                            
                            for i in range(min_t2_mp4):
                                
                                combined_video_file = combined_video_path / f"combined_1_set_{i}.mp4"
                                
                                vid1 = next(Path(take_2_gen[0], f"seq_{i}").glob("*.mp4"))         
                                vid2 = next(Path(take_2_gen[1], f"seq_{i}").glob("*.mp4"))
                                vid3 = next(Path(take_2_gen[2], f"seq_{i}").glob("*.mp4"))
                                vid4 = next(Path(take_2_gen[3], f"seq_{i}").glob("*.mp4"))
                                
                                for v in [vid1, vid2, vid3, vid4]:
                                    _ = subprocess.call([
                                        "ffmpeg", "-i", str(v), "-c", "copy", "-bsf:v", "h264_mp4toannexb", "-f", "mpegts", str(v.with_suffix(".ts"))
                                    ])

                                _ = subprocess.call([
                                    "ffmpeg", "-i", f"concat:{str(vid1.with_suffix('.ts'))}|{str(vid2.with_suffix('.ts'))}|{str(vid3.with_suffix('.ts'))}|{str(vid4.with_suffix('.ts'))}", "-c", "copy", "-bsf:a", "aac_adtstoasc", str(combined_video_file)
                                ])
                                
                                for v in [vid1, vid2, vid3, vid4]: v.with_suffix(".ts").unlink()
                            
                        print(f"END VISUALIZATION: STYLE TRANSFER {rep_i+1}/{self.config['TRAIN_PARAM']['test']['replication_times']} =====>")
                    else: print(f"END EVALUATION METRICS ONLY: STYLE TRANSFER {rep_i+1}/{self.config['TRAIN_PARAM']['test']['replication_times']} =====>")
                
                if self.emotion_control:
                    
                    metrics["emotion_control"] = []
                    data = eval_data["emotion_control"]
                    
                    rst = []
                    for actor in data.keys():
                        for take in tqdm(data[actor].keys(), desc=f"Eval Set Emotion Control {actor}"):
                            
                            z = data[actor][take]["ld_z"]
                            z_con = data[actor][take]["ld_z_con"]
                            z_emo = data[actor][take]["ld_z_emo"] 
                            z_sty = data[actor][take]["ld_z_sty"]
                            
                            attr = data[actor][take]["ld_attr"]
                            tt = take.split("_")[-1]
                            
                            allkeys = list(data[actor][take].keys())
                            z_emo_keys = [k for k in allkeys if "ld_z_emo" in k]
                            for z_emo_key in z_emo_keys:
                                
                                z_emo = data[actor][take][z_emo_key]
                                bsz = min(z.shape[0], z_con.shape[0], z_emo.shape[0], z_sty.shape[0])
                                z_con = z_con[:bsz]
                                z_emo = z_emo[:bsz]
                                z_sty = z_sty[:bsz]
                                feats_rst = self.model.diffusion_backward(bsz, z_con, z_emo, z_sty)
                                poses, trans = feats_rst["poses"], feats_rst["trans"]
                                poses = rearrange(poses, "b t j d -> b t (j d)")
                                feats_rst = torch.cat((poses,trans), dim=-1)
                                sliced_audio = data[actor][take]["ld_wav"][0:bsz*10000]
                                
                                if z_emo_key == "ld_z_emo": 
                                    take_emo = take.split("_")[-1]
                                    fetch_emo = [k for k, v in train_takes_dict.items() if f"0_{take_emo}_{take_emo}" in v][0]
                                    emo = f"original {fetch_emo}"
                                else: 
                                    take_emo = z_emo_key.split("_")[-1]
                                    fetch_emo = [k for k, v in train_takes_dict.items() if f"0_{take_emo}_{take_emo}" in v][0]
                                    emo = f"swap emo {fetch_emo} in element {self.emotion_control_take_element}"
                                info = f"{attr[0]} {attr[1]} {attr[2]} {attr[4]} yrs {tt} {emo}"
                                
                                rst.append(
                                    {
                                        "audio": sliced_audio,
                                        "feats": feats_rst,
                                        "info": info
                                    }
                                )
                                metrics["emotion_control"].append(
                                    {
                                        "actor": actor,
                                        "take": take,
                                        "swap_info": "",
                                        "rst_info": info,
                                        "src_motion": data[actor][take]["ld_motion"],
                                        "rst_motion": feats_rst,
                                        "audio": sliced_audio,
                                        "z_con": z_con,
                                        "z_emo": z_emo,
                                        "z_sty": z_sty
                                    }
                                )
                    
                    if not metrics_only:            
                        run_info = f"{self.emotion_control_actor[1:-1]}_{self.emotion_control_take_element}"
                        video_dump_r = self.model_path / "viz" / f"emotion_control_{run_info}_{self.stamp}_E{ldm_epoch}" / f"rep{rep_i}"
                        assert self.viz_type in ["CaMN"], "[LDM EVAL] Invalid viz type: [%s]" % self.viz_type
                        for i, sample_dict in enumerate(rst):
                            print(f"VISUALIZATION: EMOTION CONTROL {i} =====>")
                            video_dump = video_dump_r / f"rst_{i}"
                            # self.visualizer.animate_ldm_sample_v2(sample_dict, video_dump, self.smplx_data, self.skip_trans)
                            self.visualizer.animate_ldm_sample_v2(sample_dict, video_dump, self.smplx_data, self.skip_trans, without_txt=True)
                        
                        # if not self.EXEC_ON_CLUSTER: # workaround, manually custom_vid_concats.py if breaks on cluster
                        if True:
                            
                            print(f"VISUALIZATION: EMOTION CONTROL COMBINED =====>") 
                            
                            rst_dirs = [f for f in video_dump_r.iterdir() if f.is_dir()]
                            assert len(rst_dirs) == 64, "[LDM EVAL] Invalid number of rst dirs: [%d]" % len(rst_dirs) # 8 original emotions * 8 swapped emotions = 64
                            rst_dirs = sorted(rst_dirs, key=lambda x: int(x.name.split("_")[-1]))
                            
                            take_0_gen = rst_dirs[0:8]
                            take_1_gen = rst_dirs[8:16]
                            take_2_gen = rst_dirs[16:24]
                            take_3_gen = rst_dirs[24:32]
                            take_4_gen = rst_dirs[32:40]
                            take_5_gen = rst_dirs[40:48]
                            take_6_gen = rst_dirs[48:56]
                            take_7_gen = rst_dirs[56:64]
                            
                            min_t0_mp4 = min([len(list(gen.glob('*/*.mp4'))) for gen in take_0_gen])
                            min_t1_mp4 = min([len(list(gen.glob('*/*.mp4'))) for gen in take_1_gen])
                            min_t2_mp4 = min([len(list(gen.glob('*/*.mp4'))) for gen in take_2_gen])
                            min_t3_mp4 = min([len(list(gen.glob('*/*.mp4'))) for gen in take_3_gen])
                            min_t4_mp4 = min([len(list(gen.glob('*/*.mp4'))) for gen in take_4_gen])
                            min_t5_mp4 = min([len(list(gen.glob('*/*.mp4'))) for gen in take_5_gen])
                            min_t6_mp4 = min([len(list(gen.glob('*/*.mp4'))) for gen in take_6_gen])
                            min_t7_mp4 = min([len(list(gen.glob('*/*.mp4'))) for gen in take_7_gen])
                        
                            combined_video_path = video_dump_r / "combined"
                            combined_video_path.mkdir(parents=True, exist_ok=True)
                            
                            for ii in range(8):
                                for jj in range(eval(f"min_t{ii}_mp4")):
                                    combined_video_file = combined_video_path / f"combined_{ii}_set_{jj}.mp4"
                                    
                                    vid1 = next(Path(eval(f"take_{ii}_gen[0]"), f"seq_{jj}").glob("*.mp4"))
                                    vid2 = next(Path(eval(f"take_{ii}_gen[1]"), f"seq_{jj}").glob("*.mp4"))
                                    vid3 = next(Path(eval(f"take_{ii}_gen[2]"), f"seq_{jj}").glob("*.mp4"))
                                    vid4 = next(Path(eval(f"take_{ii}_gen[3]"), f"seq_{jj}").glob("*.mp4"))
                                    vid5 = next(Path(eval(f"take_{ii}_gen[4]"), f"seq_{jj}").glob("*.mp4"))
                                    vid6 = next(Path(eval(f"take_{ii}_gen[5]"), f"seq_{jj}").glob("*.mp4"))
                                    vid7 = next(Path(eval(f"take_{ii}_gen[6]"), f"seq_{jj}").glob("*.mp4"))
                                    vid8 = next(Path(eval(f"take_{ii}_gen[7]"), f"seq_{jj}").glob("*.mp4"))
                                    _ = subprocess.call([
                                        "ffmpeg", "-i", str(vid1), "-i", str(vid2), "-i", str(vid3), "-i", str(vid4), "-i", str(vid5), "-i", str(vid6), "-i", str(vid7), "-i", str(vid8), "-filter_complex", "[0:v][1:v]hstack[top1];[2:v][3:v]hstack[top2];[4:v][5:v]hstack[top3];[6:v][7:v]hstack[top4];[top1][top2]vstack[bottom1];[top3][top4]vstack[bottom2];[bottom1][bottom2]vstack,format=yuv420p[v]", "-map", "[v]", str(combined_video_file)
                                    ])   
                                    combined_video_file_w_audio = combined_video_path / f"combined_{ii}_set_{jj}_audio_1.mp4"
                                    _ = subprocess.call([
                                        "ffmpeg", "-i", str(combined_video_file), "-i", str(vid1), "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0", "-shortest", str(combined_video_file_w_audio)
                                    ])
                                    combined_video_file.unlink()
                                    
                                    # # old code
                                    # for kk in range(8):
                                    #     vid = next(Path(eval(f"take_{ii}_gen[{kk}]"), f"seq_{jj}").glob("*.mp4"))
                                    #     _ = subprocess.call([
                                    #         "ffmpeg", "-y", "-i", str(vid), "-c", "copy", "-bsf:v", "h264_mp4toannexb", "-f", "mpegts", str(vid.with_suffix(".ts"))
                                    #     ])

                                    # vid1 = next(Path(eval(f"take_{ii}_gen[0]"), f"seq_{jj}").glob("*.ts"))
                                    # vid2 = next(Path(eval(f"take_{ii}_gen[1]"), f"seq_{jj}").glob("*.ts"))   
                                    # vid3 = next(Path(eval(f"take_{ii}_gen[2]"), f"seq_{jj}").glob("*.ts"))   
                                    # vid4 = next(Path(eval(f"take_{ii}_gen[3]"), f"seq_{jj}").glob("*.ts"))   
                                    # vid5 = next(Path(eval(f"take_{ii}_gen[4]"), f"seq_{jj}").glob("*.ts"))   
                                    # vid6 = next(Path(eval(f"take_{ii}_gen[5]"), f"seq_{jj}").glob("*.ts"))   
                                    # vid7 = next(Path(eval(f"take_{ii}_gen[6]"), f"seq_{jj}").glob("*.ts"))   
                                    # vid8 = next(Path(eval(f"take_{ii}_gen[7]"), f"seq_{jj}").glob("*.ts"))
                                    
                                    # _ = subprocess.call([
                                    #     "ffmpeg", "-i", f"concat:{str(vid1)}|{str(vid2)}|{str(vid3)}|{str(vid4)}|{str(vid5)}|{str(vid6)}|{str(vid7)}|{str(vid8)}", "-c", "copy", "-bsf:a", "aac_adtstoasc", str(combined_video_file) 
                                    # ])
                                    
                                    # for kk in range(8):
                                    #     ts_file = next(Path(eval(f"take_{ii}_gen[{kk}]"), f"seq_{jj}").glob("*.ts"))
                                    #     _ = subprocess.call(["rm", str(ts_file)])
                    
                            print(f"VISUALIZATION: EMOTION CONTROL ORIGINAL COMBINED =====>")
                            orig_gens = [take_0_gen[0], take_1_gen[0], take_2_gen[0], take_3_gen[0], take_4_gen[0], take_5_gen[0], take_6_gen[0], take_7_gen[0]]
                            min_orig_gens_mp4 = min([len(list(gen.glob('*/*.mp4'))) for gen in orig_gens])
                            orig_gens_video_path = video_dump_r / "orig_combined"
                            orig_gens_video_path.mkdir(parents=True, exist_ok=True)
                            
                            for kk in range(min_orig_gens_mp4):
                                video_file = orig_gens_video_path / f"orig_combined_emotion_{kk}.mp4"
                                
                                vid1 = next(Path(orig_gens[0], f"seq_{kk}").glob("*.mp4"))
                                vid2 = next(Path(orig_gens[1], f"seq_{kk}").glob("*.mp4"))
                                vid3 = next(Path(orig_gens[2], f"seq_{kk}").glob("*.mp4"))
                                vid4 = next(Path(orig_gens[3], f"seq_{kk}").glob("*.mp4"))
                                vid5 = next(Path(orig_gens[4], f"seq_{kk}").glob("*.mp4"))
                                vid6 = next(Path(orig_gens[5], f"seq_{kk}").glob("*.mp4"))
                                vid7 = next(Path(orig_gens[6], f"seq_{kk}").glob("*.mp4"))
                                vid8 = next(Path(orig_gens[7], f"seq_{kk}").glob("*.mp4"))
                                _ = subprocess.call([
                                    "ffmpeg", "-i", str(vid1), "-i", str(vid2), "-i", str(vid3), "-i", str(vid4), "-i", str(vid5), "-i", str(vid6), "-i", str(vid7), "-i", str(vid8), "-filter_complex", "[0:v][1:v]hstack[top1];[2:v][3:v]hstack[top2];[4:v][5:v]hstack[top3];[6:v][7:v]hstack[top4];[top1][top2]vstack[bottom1];[top3][top4]vstack[bottom2];[bottom1][bottom2]vstack,format=yuv420p[v]", "-map", "[v]", str(video_file)
                                ])
                                video_file_w_audio = orig_gens_video_path / f"orig_combined_emotion_{kk}_audio_1.mp4"
                                _ = subprocess.call([
                                    "ffmpeg", "-i", str(video_file), "-i", str(vid1), "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0", "-shortest", str(video_file_w_audio)
                                ])
                                video_file.unlink()
                                
                                # for ll in range(8):
                                #     vid = next(Path(orig_gens[ll], f"seq_{kk}").glob("*.mp4"))
                                #     _ = subprocess.call([
                                #         "ffmpeg", "-y", "-i", str(vid), "-c", "copy", "-bsf:v", "h264_mp4toannexb", "-f", "mpegts", str(vid.with_suffix(".ts"))
                                #     ])
                                # vid1 = next(Path(orig_gens[0], f"seq_{kk}").glob("*.ts"))
                                # vid2 = next(Path(orig_gens[1], f"seq_{kk}").glob("*.ts"))
                                # vid3 = next(Path(orig_gens[2], f"seq_{kk}").glob("*.ts"))
                                # vid4 = next(Path(orig_gens[3], f"seq_{kk}").glob("*.ts"))
                                # vid5 = next(Path(orig_gens[4], f"seq_{kk}").glob("*.ts"))
                                # vid6 = next(Path(orig_gens[5], f"seq_{kk}").glob("*.ts"))
                                # vid7 = next(Path(orig_gens[6], f"seq_{kk}").glob("*.ts"))
                                # vid8 = next(Path(orig_gens[7], f"seq_{kk}").glob("*.ts"))
                                # _ = subprocess.call([
                                #     "ffmpeg", "-i", f"concat:{str(vid1)}|{str(vid2)}|{str(vid3)}|{str(vid4)}|{str(vid5)}|{str(vid6)}|{str(vid7)}|{str(vid8)}", "-c", "copy", "-bsf:a", "aac_adtstoasc", str(video_file) 
                                # ])
                                # for zz in range(8):
                                #     ts_file = next(Path(orig_gens[zz], f"seq_{kk}").glob("*.ts"))
                                #     _ = subprocess.call(["rm", str(ts_file)])
                        
                        print(f"END VISUALIZATION: EMOTION CONTROL {rep_i+1}/{self.config['TRAIN_PARAM']['test']['replication_times']} =====>")
                    else: print(f"END EVALUATION METRICS ONLY: EMOTION CONTROL {rep_i+1}/{self.config['TRAIN_PARAM']['test']['replication_times']} =====>")

    def _dump_args(self):
        if not self.debug:
            if self.tag == "diffusion":
                arch_cfg_name = self.config["TRAIN_PARAM"][self.tag]["arch"] + ".json"
                arch_cfg = self.model_path_r.parent / "configs" / arch_cfg_name
                with open(arch_cfg, "r") as f: arch_cfg = json.load(f)
                self.cfg_dump =  {**self.config, **arch_cfg} 
            elif self.tag in ["motionprior", "motionprior_long"]:
                if self.config["TRAIN_PARAM"]["motionprior"]["emotional"]: 
                    if "_fing" in self.config["TRAIN_PARAM"]["diffusion"]["lmdb_cache"]: dmp_cfg_name = "prior_emotional_fing"
                    else: dmp_cfg_name = "prior_emotional"
                else: dmp_cfg_name = "prior"
                with open(str(Path(self.processed.parents[1], f"configs/{dmp_cfg_name}.json")), "r") as f: arch_cfg = json.load(f)
                self.cfg_dump =  {**self.config, **arch_cfg} 
            else: self.cfg_dump = self.config
            with open(self.model_path / "experiment_args.json", "w") as f:
                json.dump(self.cfg_dump, f, indent=4)
                
 