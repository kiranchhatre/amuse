
import json
import torch
import inspect
import diffusers
from tqdm import tqdm
import torch.nn as nn
from pathlib import Path

from models.latent_diffusion.infer_pretrained_vae import PretrainedVAE
from models.latent_diffusion.denoiser import Denoiser

class LatentDiffusionModel(nn.Module):
    
    def __init__(self) -> None:
        super().__init__() 

    def setup(self, processed, config, device, tag, vae_base_arch, base_con_ae, base_emo_ae, base_audio_ae, backup_cfg, combined_train=False):
        
        self.tag = tag
        self.config = config
        self.device = device
        self.processed = processed
        self.combined_train = combined_train
        self.smplx_data = self.config["TRAIN_PARAM"]["latent_diffusion"]["smplx_data"]
        self.skip_trans = self.config["TRAIN_PARAM"]["latent_diffusion"]["skip_trans"]
        self.train_upper_body = self.config["TRAIN_PARAM"]["latent_diffusion"]["train_upper_body"]
        self.smplx_rep = self.config["TRAIN_PARAM"]["latent_diffusion"]["smplx_rep"]
        if not self.combined_train:
            self.pretrained_vae = PretrainedVAE(self.device)
            self.pretrained_vae.load_model(self.config, self.processed, self.tag, vae_base_arch, backup_cfg)      
        
        ldm_arch = self.config["TRAIN_PARAM"]["latent_diffusion"]["arch"]
        with open(str(Path(self.processed.parents[1], f"configs/{ldm_arch}.json")), "r") as f:
            self.ldm_cfg = json.load(f)
        self.num_inference_timesteps = self.ldm_cfg["scheduler"]["num_inference_timesteps"]
        self.eta = self.ldm_cfg["scheduler"]["eta"]
        self.latent_dim = self.ldm_cfg["arch_denoiser"]["latent_dim"]
        self.seq_len = self.config["DATA_PARAM"]["Bvh"]["train_pose_framelen"]
        
        self.noise_scheduler = diffusers.DDPMScheduler(
            num_train_timesteps=self.ldm_cfg["noisy_scheduler"]["num_train_timesteps"],
            beta_start=self.ldm_cfg["noisy_scheduler"]["beta_start"],
            beta_end=self.ldm_cfg["noisy_scheduler"]["beta_end"],
            beta_schedule=self.ldm_cfg["noisy_scheduler"]["beta_schedule"],
            variance_type=self.ldm_cfg["noisy_scheduler"]["variance_type"],
            clip_sample=self.ldm_cfg["noisy_scheduler"]["clip_sample"],
            prediction_type=self.ldm_cfg["noisy_scheduler"]["prediction_type"],
        )
        self.scheduler = diffusers.DDIMScheduler(
            num_train_timesteps=self.ldm_cfg["scheduler"]["num_train_timesteps"],
            beta_start=self.ldm_cfg["scheduler"]["beta_start"],
            beta_end=self.ldm_cfg["scheduler"]["beta_end"],
            beta_schedule=self.ldm_cfg["scheduler"]["beta_schedule"],
            set_alpha_to_one=self.ldm_cfg["scheduler"]["set_alpha_to_one"],
            steps_offset=self.ldm_cfg["scheduler"]["steps_offset"],
        )
        denoiser_cfg = self.ldm_cfg["arch_denoiser"]
        if self.smplx_data: denoiser_cfg["smplx_data"] = self.smplx_data
        if self.skip_trans: denoiser_cfg["skip_trans"] = self.skip_trans
        if self.train_upper_body: denoiser_cfg["train_upper_body"] = self.train_upper_body
        denoiser_cfg["smplx_rep"] = self.smplx_rep
        self.denoiser = Denoiser(denoiser_cfg)
    
    def infer_pretrained_vae(self, batch, stage):
        motion_batch = batch["ld_motion"]
        if stage == "encode": return self.pretrained_vae.get_latent(motion_batch)
        elif stage == "decode": pass
        else: raise ValueError(f"[LDM] Invalid stage: {stage} for infer_pretrained_vae")
    
    def diffusion_forward(self, z, ld_audio_con, ld_audio_emo, ld_audio_sty, plot_latent=False, 
                          emo_label=None, plot_path=None, attr=None, lengths=None, ld_audio_mfcc=None):
        
        z = z.permute(1, 0, 2)
        noise = torch.randn_like(z)
        bsz = z.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz, ),
            device=self.device,
        )
        timesteps = timesteps.long()
        noisy_latents = self.noise_scheduler.add_noise(z.clone(), noise, timesteps)
        if ld_audio_mfcc is None:
            z_con = ld_audio_con[:, None, :]
            z_emo = ld_audio_emo[:, None, :] if ld_audio_emo is not None else None
            z_sty = ld_audio_sty[:, None, :] if ld_audio_sty is not None else None
        else: raise NotImplementedError("LPDM: Baseline audio AE not implemented yet")
        
        noise_pred = self.denoiser(
            sample=noisy_latents,
            timestep=timesteps,
            con_hidden=z_con,
            emo_hidden=z_emo,
            sty_hidden=z_sty,
            lengths=lengths,
            return_dict=False,
        )[0]
        
        if self.ldm_cfg["losses"]["LAMBDA_PRIOR"] != 0:
            noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
            noise, noise_prior = torch.chunk(noise, 2, dim=0)
        else:
            noise_pred_prior = 0
            noise_prior = 0
        n_set = {
            "noise": noise,
            "noise_prior": noise_prior,
            "noise_pred": noise_pred,
            "noise_pred_prior": noise_pred_prior,
        }    
        if not self.ldm_cfg["losses"]["predict_epsilon"]:
            n_set["pred"] = noise_pred
            n_set["latent"] = z
        return n_set
            
    def diffusion_backward(self, ld_audio_con, ld_audio_emo, ld_audio_sty, ld_audio_mfcc, bsz):
        
        if ld_audio_mfcc is None:
            z_con = ld_audio_con[:, None, :]
            z_emo = ld_audio_emo[:, None, :] if ld_audio_emo is not None else None
            z_sty = ld_audio_sty[:, None, :] if ld_audio_sty is not None else None   
        else: raise NotImplementedError("LPDM: Baseline audio AE not implemented yet")
        
        lengths_reverse = [self.seq_len] * bsz   
        latents = torch.randn(
            (bsz, self.latent_dim[0], self.latent_dim[-1]),
             device=self.device,
             dtype=torch.float
        )
        latents = latents * self.scheduler.init_noise_sigma
        self.scheduler.set_timesteps(self.num_inference_timesteps)
        timesteps = self.scheduler.timesteps.to(self.device)
        extra_step_kwargs = {}
        if "eta" in set(inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = self.eta
        
        for _, t in enumerate(timesteps):
            latent_model_input = latents
            noise_pred = self.denoiser(
                sample=latent_model_input,
                timestep=t,
                con_hidden=z_con,
                emo_hidden=z_emo,
                sty_hidden=z_sty,
                lengths=lengths_reverse,
            )[0]
            latents = self.scheduler.step(noise_pred, t, latents, 
                                          **extra_step_kwargs).prev_sample
        
        latents = latents.permute(1, 0, 2)
        return latents
   