
import re
import json
import torch
import inspect
import pickle
import diffusers
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
import torchaudio.transforms as T
import torch.nn.functional as F
from einops import rearrange, repeat
from pytorch3d import transforms as p3d_tfs

from dm.utils.ldm_evals import *
from dm.utils.wav_utils import *
from models import Pretrained_AST_EVP
from models.latent_diffusion.denoiser import Denoiser
from models.latent_diffusion.infer_pretrained_vae import PretrainedVAE

class PretrainedLPDM_v1():
    def __init__(self, base_prior, base_con_ae=None, base_emo_ae=None, base_audio_ae=None):
        self.base_vae = base_prior
        self.con_ae = base_con_ae
        self.emo_ae = base_emo_ae
        self.base_ae = base_audio_ae

    def setup(self, config, device, processed, backup_cfg, EXEC_ON_CLUSTER, baseline=False, verbose=False, diffonly=False):
        
        self.config = config
        self.device = device
        self.processed = processed
        self.backup_cfg = backup_cfg
        self.baseline = baseline
        self.diffonly = diffonly
        self.tag = self.config["TRAIN_PARAM"]["tag"]
        self.smplx_data = self.config["TRAIN_PARAM"]["latent_diffusion"]["smplx_data"]
        self.smplx_rep = self.config["TRAIN_PARAM"]["latent_diffusion"]["smplx_rep"]
        self.skip_trans = self.config["TRAIN_PARAM"]["latent_diffusion"]["skip_trans"]
        self.train_upper_body = self.config["TRAIN_PARAM"]["latent_diffusion"]["train_upper_body"]
        if self.train_upper_body: self.lower_body_jts = [1, 2, 4, 5, 7, 8, 10, 11]
        self.style_transfer = self.config["TRAIN_PARAM"]["test"]["style_transfer"]["use"]
        self.emotion_control = self.config["TRAIN_PARAM"]["test"]["emotion_control"]["use"]
        self.content_control = self.config["TRAIN_PARAM"]["test"]["content_control"]["use"]
        self.style_Xemo_transfer = self.config["TRAIN_PARAM"]["test"]["style_Xemo_transfer"]["use"]
        self.train_pose_framelen = self.config["DATA_PARAM"]["Bvh"]["train_pose_framelen"]
        self.target_length = self.config["TRAIN_PARAM"]["wav_dtw_mfcc"]["target_length"]
        self.norm_mean = self.config["TRAIN_PARAM"]["wav_dtw_mfcc"]["dataset_mean"]
        self.norm_std = self.config["TRAIN_PARAM"]["wav_dtw_mfcc"]["dataset_std"]
        self.mfcc_transform = T.MFCC(sample_rate=config["DATA_PARAM"]["Wav"]["sample_rate"], 
                                n_mfcc=config["DATA_PARAM"]["Wav"]["n_mfcc"], 
                                melkwargs={"n_fft": config["DATA_PARAM"]["Wav"]["n_fft"],
                                        "n_mels": config["DATA_PARAM"]["Wav"]["n_mels"],
                                        "hop_length": config["DATA_PARAM"]["Wav"]["hop_length"],
                                        "mel_scale": config["DATA_PARAM"]["Wav"]["mel_scale"],},)
        
        lpdm_cfg = {
            "saved_model_dir": "saved-models" if not EXEC_ON_CLUSTER else "saved-models-new",
            "load_epoch_prior": self.config["TRAIN_PARAM"]["latent_diffusion"]["pretrained_prior_lpdm_e"],
            "load_epoch_ldm": self.config["TRAIN_PARAM"]["latent_diffusion"]["pretrained_ldm_lpdm_e"],
        }
        assert lpdm_cfg["load_epoch_prior"] == lpdm_cfg["load_epoch_ldm"], "Epochs for prior and ldm should be same"
        
        ldm_arch = self.config["TRAIN_PARAM"]["latent_diffusion"]["arch"]
        with open(str(Path(self.processed.parents[1], f"configs/{ldm_arch}.json")), "r") as f: self.ldm_cfg = json.load(f)
        denoiser_cfg = self.ldm_cfg["arch_denoiser"]
        if self.smplx_data: denoiser_cfg["smplx_data"] = self.smplx_data
        if self.skip_trans: denoiser_cfg["skip_trans"] = self.skip_trans
        if self.train_upper_body: denoiser_cfg["train_upper_body"] = self.train_upper_body
        denoiser_cfg["smplx_rep"] = self.smplx_rep
        self.denoiser = Denoiser(denoiser_cfg)
        
        pretrained_lpdm = self.config["TRAIN_PARAM"]["latent_diffusion"]["pretrained_lpdm"]
        saved_model_path = processed.parents[1] / lpdm_cfg["saved_model_dir"] / pretrained_lpdm
        if self.backup_cfg is not None: raise NotImplementedError("Backup for LPDM not implemented yet!")
        lpdm_models = [f for f in saved_model_path.iterdir() if f.is_file() and "experiment_args.json" not in str(f) and f.stem.split("_")[0] == "latdiff"]
        if lpdm_cfg["load_epoch_ldm"] == "best":
            total_L = np.inf
            for lpdm_model in lpdm_models:
                tL = float(re.findall("\d+\.\d+", lpdm_model.stem.split("_")[-2])[0])
                if tL < total_L: total_L = tL; best_ldm_model = lpdm_model
        else: best_ldm_model = [f for f in lpdm_models if int(re.search(r'\d+', f.stem.split("_")[-1]).group()) == int(lpdm_cfg["load_epoch_ldm"])][0]
        ldm_epoch = int(re.search(r'\d+', best_ldm_model.stem.split("_")[-1]).group())
        chkpt = torch.load(best_ldm_model) 
        chkpt_modules = []
        for k in chkpt['model_state_dict'].keys(): chkpt_modules.append(k.split(".")[0]) if k.split(".")[0] not in chkpt_modules else None
        print("[LDM] <===== Chosen LDM model based on total loss: ", best_ldm_model, " with modules: ", set(chkpt_modules), " =====>")
        
        tgt_denoiser = self.denoiser.state_dict()
        modules = {"denoiser": tgt_denoiser}
        for module, tgt_state in modules.items():
            state_dict_count = 0
            for name, param in chkpt['model_state_dict'].items():
                if name.startswith(module):
                    state_dict_count += 1
                    name = name[len(module)+1:]
                    assert isinstance(param, nn.Parameter) or isinstance(param, torch.Tensor), f"param {param} is not a nn.Parameter or torch.Tensor"
                    param = param.data
                    assert name in tgt_state, f"key {name} not found in {module}"
                    tgt_state[name].copy_(param)
            assert state_dict_count == len(tgt_state), f"state_dict_count {state_dict_count} != len(tgt_state) {len(tgt_state)} for {module}"
            print(f"[LATDIFF EVAL] <===== {module} loaded, ", state_dict_count, " state_dicts =====>")
        self.denoiser = self._prepare_model(self.denoiser, tgt_denoiser, verbose)
        
        self.pretrained_vae = PretrainedVAE(self.device)
        if lpdm_cfg["load_epoch_prior"] == "best": lpdm_cfg["load_epoch_prior"] = ldm_epoch
        self.pretrained_vae.load_model(self.config, self.processed, "latent_diffusion", self.base_vae, backup_cfg, lpdm_cfg)  
        
        self.dtw = Pretrained_AST_EVP(device)
        if "ablation" in config["TRAIN_PARAM"]["wav_dtw_mfcc"]: audio_ablation = self.config['TRAIN_PARAM']['wav_dtw_mfcc']['ablation']
        assert audio_ablation is not None, f"[LPDM EVAL] Audio ablation flag: {audio_ablation}"
        self.dtw.get_model(self.config, self.processed, self.tag, audio_ablation=audio_ablation)   
        
        self.scheduler = diffusers.DDIMScheduler(
            num_train_timesteps=self.ldm_cfg["scheduler"]["num_train_timesteps"],
            beta_start=self.ldm_cfg["scheduler"]["beta_start"],
            beta_end=self.ldm_cfg["scheduler"]["beta_end"],
            beta_schedule=self.ldm_cfg["scheduler"]["beta_schedule"],
            set_alpha_to_one=self.ldm_cfg["scheduler"]["set_alpha_to_one"],
            steps_offset=self.ldm_cfg["scheduler"]["steps_offset"],
        )
        self.num_inference_timesteps = self.ldm_cfg["scheduler"]["num_inference_timesteps"]
        self.eta = self.ldm_cfg["scheduler"]["eta"]
        self.latent_dim = self.ldm_cfg["arch_denoiser"]["latent_dim"]
        self.seq_len = self.config["DATA_PARAM"]["Bvh"]["train_pose_framelen"]
        return ldm_epoch
    
    def diffusion_backward(self, bsz, z_con, z_emo, z_sty):
        z_con = z_con[:, None, :]
        z_emo = z_emo[:, None, :] if z_emo is not None else None
        z_sty = z_sty[:, None, :] if z_sty is not None else None
        
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
        
        with torch.no_grad():
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
        
        if not self.diffonly: 
            motionfeats = self.pretrained_vae.get_motion(latents, lengths_reverse)
            motion_3D = dict()
            if self.smplx_rep == "6D":
                rot6D, trans = motionfeats[:, :, :-3], motionfeats[:, :, -3:]
                rot6D = rearrange(rot6D, "b t (j r) -> b t j r", r=6)
                mat = p3d_tfs.rotation_6d_to_matrix(rot6D)
                poses = p3d_tfs.matrix_to_axis_angle(mat)
                motion_3D["poses"], motion_3D["trans"] = poses, trans
            else:
                motion_3D["poses"], motion_3D["trans"] = motionfeats[:, :, :-3], motionfeats[:, :, -3:]
                motion_3D["poses"] = rearrange(motion_3D["poses"], "b t (j r) -> b t j r", r=3)
        else: raise # motion_3D = self.motionenc.dec(latents.squeeze()[None, :, None, None])[None, :, :]
        return motion_3D
    
    def process_single_seq(self, sliced_chunk, framerate=16000//2, baseline=False):
        
        fbank = torchaudio.compliance.kaldi.fbank(sliced_chunk, htk_compat=True, sample_frequency=16000, use_energy=False, window_type='hanning', 
                                                num_mel_bins=self.config["TRAIN_PARAM"]["wav_dtw_mfcc"]["num_mel_bins"], dither=0.0, frame_shift=10)
        n_frames = fbank.shape[0]
        p = self.target_length - n_frames
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0: fbank = fbank[0:self.target_length, :]
        fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        ast_feats = self.dtw.get_features(fbank.unsqueeze(0)) # with batch dummy dim
        con, emo, sty = ast_feats["con"].squeeze(0), ast_feats["emo"].squeeze(0), ast_feats["sty"].squeeze(0)
        return con[None, :], emo[None, :], sty[None, :]
    
    def collect_audio_metrics(self, sliced_chunk, framerate=16000//2, baseline=False, tgtpath=None):
        fbank = torchaudio.compliance.kaldi.fbank(sliced_chunk, htk_compat=True, sample_frequency=16000, use_energy=False, window_type='hanning', 
                                                num_mel_bins=self.config["TRAIN_PARAM"]["wav_dtw_mfcc"]["num_mel_bins"], dither=0.0, frame_shift=10)
        n_frames = fbank.shape[0]
        p = self.target_length - n_frames
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0: fbank = fbank[0:self.target_length, :]
        fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        fbank_dict = self.dtw.get_reconstructed_fbank(fbank.unsqueeze(0)) 
        metric_dump = tgtpath / "audio_metrics"
        metric_dump.mkdir(parents=True, exist_ok=True)
        with open(metric_dump / "fbank.pkl", "wb") as f: pickle.dump(fbank_dict, f)
    
    def _process_baseline_single_seq(self, audio, framerate=16000//2):
        current_audio = audio.set_frame_rate(framerate)
        channel_sounds = current_audio.split_to_mono()
        samples = [s.get_array_of_samples() for s in channel_sounds]
        fp_arr = np.array(samples).T.astype(np.float32)
        fp_arr /= np.iinfo(samples[0].typecode).max 
        fp_arr = F.pad(torch.from_numpy(fp_arr), (0, 0, 0, 160000 - fp_arr.shape[0]), value=0.0)
        audio_mfcc = audio2slicedmfcc(self.config, batch_raw_wf=fp_arr[None, ...]).squeeze()
        audio_batch = audio_mfcc[None, None, :, :]
        audio_batch = audio_batch.to(self.device) if not audio_batch.is_cuda else audio_batch
        with torch.no_grad():
            z_base, _ = self.base_ae(audio_batch)
            z_base = z_base.squeeze()
        return z_base[None, :], None

    def process_loader(self, data_dict):
        
        """
        Broad combinations:
        ===

        1 Within one person, swap emotion latent only, keeping content and style the same: This would result in the same 
        utterance being spoken in different emotions. It's a straightforward way to change the emotional tone of the same 
        content.

        2 Within two persons, swap style vectors and emotion latents for the same utterances: This represents style transfer
        and emotion swap between two different people. This could lead to the same content spoken in the style and emotion 
        of another person.

        3 Within two persons, with different utterances and different emotions, swap style latent and emotion latent: This 
        combination involves changing both emotion and style across different content, resulting in a style transfer to a 
        different emotion of the target person. It's a more complex manipulation but can lead to interesting results.

        Additional:
        ===
        4 Within one person, swap content latent only, keeping emotion and style the same: This would result in the same 
        emotion and style speaking different content. It could be useful for creating variations of the same style and 
        emotion.

        5 Within one person, swap style latent only, keeping emotion and content the same: This would result in the same 
        content and emotion spoken in a different style. It's similar to combination 1 but focused on style.

        6 Within two persons, swap emotion latent only for the same content: This would result in the same content spoken 
        in different emotions by different people, preserving the style.

        7 Within one person, keep content and style the same but interpolate between different emotion latents: This could 
        result in a gradual emotional transition within the same content and style.

        8 Combine style transfer and emotion transfer within one person: Swap style and emotion latents, resulting in a 
        different emotion and style for the same content. This can create interesting combinations.

        
        """

        loader_data = dict()
        
        if self.style_Xemo_transfer: 
            info = data_dict["style_Xemo_transfer_info"]
            data = data_dict["style_Xemo_transfer"]

            if "," not in info: 
                # [lu-lawrence]_[angry-happy]_                                  Edits
                # *lu_angry_0_73_73 : a1 t1                                     lawrence happy 
                # *lu_happy_0_65_65 : a1 t2                                     lawrence angry  
                # *lawrence_angry_0_73_73 : a2 t3                               lu happy    
                # *lawrence_happy_0_65_65* : a2 t4                              lu angry
                actor1, actor2 = info.split("_")[0][1:-1].split("-")[0], info.split("_")[0][1:-1].split("-")[1]
                take1, take2, take3, take4 = "_".join(info.split("*")[1].split("_")[2:]), "_".join(info.split("*")[2].split("_")[2:]),\
                                             "_".join(info.split("*")[3].split("_")[2:]), "_".join(info.split("*")[4].split("_")[2:])
                assert all([take1 == take3, take2 == take4]), "Takes are not the same for style transfer!"
                
                assert data[actor1][take1]["ld_emo_label"] == data[actor2][take1]["ld_emo_label"], f"Emotion labels are not the same for style transfer! {data[actor1][take1]['ld_emo_label']} != {data[actor2][take1]['ld_emo_label']}"
                assert data[actor1][take2]["ld_emo_label"] == data[actor2][take2]["ld_emo_label"], f"Emotion labels are not the same for style transfer! {data[actor1][take2]['ld_emo_label']} != {data[actor2][take2]['ld_emo_label']}"
                
                actor1_motion1 = torch.from_numpy(data[actor1][take1]["ld_motion"]).to(self.device)
                actor1_audio1 = data[actor1][take1]["ld_waveform"]
                actor2_motion1 = torch.from_numpy(data[actor2][take3]["ld_motion"]).to(self.device)
                actor2_audio1 = data[actor2][take3]["ld_waveform"]
                actor1_motion2 = torch.from_numpy(data[actor1][take2]["ld_motion"]).to(self.device)
                actor1_audio2 = data[actor1][take2]["ld_waveform"]
                actor2_motion2 = torch.from_numpy(data[actor2][take4]["ld_motion"]).to(self.device)
                actor2_audio2 = data[actor2][take4]["ld_waveform"]
                
                if self.skip_trans: raise NotImplementedError("Skip trans for style_Xemo_transfer not implemented yet!")
                elif self.train_upper_body: raise NotImplementedError("Train upper body for style_Xemo_transfer not implemented yet!")
                
                for actor, take, motion, audio in tqdm([(actor1, take1, actor1_motion1, actor1_audio1), 
                                                        (actor2, take3, actor2_motion1, actor2_audio1), 
                                                        (actor1, take2, actor1_motion2, actor1_audio2), 
                                                        (actor2, take4, actor2_motion2, actor2_audio2)], 
                                                        desc="Processing style Xemo transfer latents", leave=False):
                    z = self._loader_helper_v1(motion, audio)
                    data[actor][take]["ld_z"] = z["z_motion"]
                    data[actor][take]["ld_z_con"] = z["z_con"]
                    data[actor][take]["ld_z_emo"] = z["z_emo"]
                    data[actor][take]["ld_z_sty"] = z["z_sty"]
                
                # swap emotion and styles
                data[actor1][take1][f"ld_z_emo_{actor2}_{take4}"] = data[actor2][take4]["ld_z_emo"]
                data[actor1][take1][f"ld_z_sty_{actor2}_{take4}"] = data[actor2][take4]["ld_z_sty"]
                
                data[actor2][take3][f"ld_z_emo_{actor1}_{take2}"] = data[actor1][take2]["ld_z_emo"]
                data[actor2][take3][f"ld_z_sty_{actor1}_{take2}"] = data[actor1][take2]["ld_z_sty"]
                
                data[actor1][take2][f"ld_z_emo_{actor2}_{take3}"] = data[actor2][take3]["ld_z_emo"]
                data[actor1][take2][f"ld_z_sty_{actor2}_{take3}"] = data[actor2][take3]["ld_z_sty"]

                data[actor2][take4][f"ld_z_emo_{actor1}_{take1}"] = data[actor1][take1]["ld_z_emo"]
                data[actor2][take4][f"ld_z_sty_{actor1}_{take1}"] = data[actor1][take1]["ld_z_sty"]
                
                data["takes"] = f"{take1}*{take2}*{take3}*{take4}"
                
                loader_data["style_Xemo_transfer"] = data
            
            else: raise NotImplementedError("Multiple style transfer not implemented yet")
        
        if self.style_transfer:
            info = data_dict["style_transfer_info"]
            data = data_dict["style_transfer"]

            if "," not in info: # [ayana-scott]_[fear]
                actor1, actor2 = info.split("_")[0][1:-1].split("-")[0], info.split("_")[0][1:-1].split("-")[1]
                takes = mapinfo2takes(info)
                take1, take2 = takes[0], takes[1]
                
                assert data[actor1][take1]["ld_emo_label"] == data[actor2][take1]["ld_emo_label"] == \
                       data[actor1][take2]["ld_emo_label"] == data[actor2][take2]["ld_emo_label"], \
                       "Emotion labels are not the same for style transfer!"

                actor1_motion1 = torch.from_numpy(data[actor1][take1]["ld_motion"]).to(self.device)
                actor1_audio1 = data[actor1][take1]["ld_waveform"]
                actor2_motion1 = torch.from_numpy(data[actor2][take1]["ld_motion"]).to(self.device)
                actor2_audio1 = data[actor2][take1]["ld_waveform"]
                actor1_motion2 = torch.from_numpy(data[actor1][take2]["ld_motion"]).to(self.device)
                actor1_audio2 = data[actor1][take2]["ld_waveform"]
                actor2_motion2 = torch.from_numpy(data[actor2][take2]["ld_motion"]).to(self.device)
                actor2_audio2 = data[actor2][take2]["ld_waveform"]
                
                if self.skip_trans:
                    actor1_motion1 = actor1_motion1[:, :-3]
                    actor2_motion1 = actor2_motion1[:, :-3]
                    actor1_motion2 = actor1_motion2[:, :-3]
                    actor2_motion2 = actor2_motion2[:, :-3]
                elif self.train_upper_body:
                    actor1_motion1 = self._prepare_upper_body_data(actor1_motion1)
                    actor2_motion1 = self._prepare_upper_body_data(actor2_motion1)
                    actor1_motion2 = self._prepare_upper_body_data(actor1_motion2)
                    actor2_motion2 = self._prepare_upper_body_data(actor2_motion2)
                
                for actor, take, motion, audio in tqdm([(actor1, take1, actor1_motion1, actor1_audio1), 
                                                        (actor2, take1, actor2_motion1, actor2_audio1), 
                                                        (actor1, take2, actor1_motion2, actor1_audio2), 
                                                        (actor2, take2, actor2_motion2, actor2_audio2)], 
                                                        desc="Processing style transfer latents", leave=False):
                    z = self._loader_helper_v1(motion, audio)
                    data[actor][take]["ld_z"] = z["z_motion"]
                    data[actor][take]["ld_z_con"] = z["z_con"]
                    data[actor][take]["ld_z_emo"] = z["z_emo"]
                    data[actor][take]["ld_z_sty"] = z["z_sty"]
                
                # swap styles and emotions
                data[actor1][take1][f"ld_z_sty_{actor2}"] = data[actor2][take1]["ld_z_emo"]
                data[actor1][take1][f"ld_z_emo_{actor2}"] = data[actor2][take1]["ld_z_sty"]
                
                data[actor2][take1][f"ld_z_sty_{actor1}"] = data[actor1][take1]["ld_z_emo"]
                data[actor2][take1][f"ld_z_emo_{actor1}"] = data[actor1][take1]["ld_z_sty"]
                
                data[actor1][take2][f"ld_z_sty_{actor2}"] = data[actor2][take2]["ld_z_emo"]
                data[actor1][take2][f"ld_z_emo_{actor2}"] = data[actor2][take2]["ld_z_sty"]
                
                data[actor2][take2][f"ld_z_sty_{actor1}"] = data[actor1][take2]["ld_z_emo"]
                data[actor2][take2][f"ld_z_emo_{actor1}"] = data[actor1][take2]["ld_z_sty"]
                
                loader_data["style_transfer"] = data
            
            else: raise NotImplementedError("Multiple style transfer not implemented yet")
        
        if self.emotion_control:
            info = data_dict["emotion_control_info"]
            data = data_dict["emotion_control"]
            
            if "," not in info: # [yingqing]_[happy]_first
                for actor in data.keys():
                    for take in tqdm(data[actor].keys(), desc="Processing emotion control latents", leave=False): 
                        # ld_motion, ld_wav, ld_emo_label, ld_attr, ld_z, ld_z_con, ld_z_emo
                        motion = torch.from_numpy(data[actor][take]["ld_motion"]).to(self.device)
                        if self.skip_trans: motion = motion[:, :-3]
                        elif self.train_upper_body: motion = self._prepare_upper_body_data(motion)
                        z = self._loader_helper_v1(motion, data[actor][take]["ld_waveform"])
                        data[actor][take]["ld_z"] = z["z_motion"]
                        data[actor][take]["ld_z_con"] = z["z_con"]
                        data[actor][take]["ld_z_emo"] = z["z_emo"]
                        data[actor][take]["ld_z_sty"] = z["z_sty"]
            
                for actor in data.keys():
                    for take in data[actor].keys():
                        for other_take in data[actor].keys():
                            if other_take != take:
                                data[actor][take][f"ld_z_emo_{other_take}"] = data[actor][other_take]["ld_z_emo"]
                
                loader_data["emotion_control"] = data
            
            else: raise NotImplementedError("Emotion control with multiple actors or multiple content emotions not implemented yet")    
        
        return loader_data   

    def _loader_helper_v1(self, motion, audio):
        
        if not self.baseline:
            total_chunks = audio.shape[1]//160000
            con, emo, sty = [], [], []
            for k in range(0, total_chunks):
                sliced_chunk = audio[:, k:k+160000]
                fbank = torchaudio.compliance.kaldi.fbank(sliced_chunk, htk_compat=True, sample_frequency=16000, use_energy=False, window_type='hanning', 
                                                        num_mel_bins=self.config["TRAIN_PARAM"]["wav_dtw_mfcc"]["num_mel_bins"], dither=0.0, frame_shift=10)
                n_frames = fbank.shape[0]
                p = self.target_length - n_frames
                if p > 0:
                    m = torch.nn.ZeroPad2d((0, 0, 0, p))
                    fbank = m(fbank)
                elif p < 0: fbank = fbank[0:self.target_length, :]
                fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
                ast_feats = self.dtw.get_features(fbank.unsqueeze(0)) # with batch dummy dim
                con.append(ast_feats["con"].squeeze(0))
                emo.append(ast_feats["emo"].squeeze(0))
                sty.append(ast_feats["sty"].squeeze(0)) 
            audio_con = torch.stack(con, dim=0)
            audio_emo = torch.stack(emo, dim=0)
            audio_sty = torch.stack(sty, dim=0)
            
        else: raise NotImplementedError
        # # audio_batch = []
        # # for pp in range(motion.shape[0] // self.train_pose_framelen):
        # #     current_audio = audio[pp*10000:(pp+1)*10000]
        # #     current_audio = current_audio.set_frame_rate(16000)
        # #     channel_sounds = current_audio.split_to_mono()
        # #     samples = [s.get_array_of_samples() for s in channel_sounds]
        # #     fp_arr = np.array(samples).T.astype(np.float32)
        # #     fp_arr /= np.iinfo(samples[0].typecode).max 
        # #     fp_arr = F.pad(torch.from_numpy(fp_arr), (0, 0, 0, 160000 - fp_arr.shape[0]), value=0.0)
        # #     audio_mfcc = audio2slicedmfcc(self.config, batch_raw_wf=fp_arr[None, ...])
        # #     audio_batch.append(audio_mfcc.squeeze())
        # # audio_batch = torch.stack(audio_batch)[:, None, ...]
        
        motion_batch = []
        for pp in range(motion.shape[0] // self.train_pose_framelen):
            motion_batch.append(motion[pp*self.train_pose_framelen:(pp+1)*self.train_pose_framelen])
        motion_batch = torch.stack(motion_batch)
        if self.smplx_rep == "6D":
            poses, trans = motion_batch[:, :, :-3], motion_batch[:, :, -3:]
            poses = rearrange(poses, "b s (j c) -> b s j c", j=55, c=3)
            mat = p3d_tfs.axis_angle_to_matrix(poses)   
            rot6D = p3d_tfs.matrix_to_rotation_6d(mat) 
            rot6D = rearrange(rot6D, "b s j c -> b s (j c)")
            motion_batch = torch.cat((rot6D, trans), dim=-1) # bs, seq, 333
        z_motion_batch = self.pretrained_vae.get_latent(motion_batch).squeeze()
        
        # # if not self.baseline:
        # #     con_batch, emo_batch, sty_batch = [], [], []
        # #     for pp in range(audio_con.shape[0] // self.train_pose_framelen):
        # #         con_batch.append(audio_con[pp*self.train_pose_framelen:(pp+1)*self.train_pose_framelen])
        # #         emo_batch.append(audio_emo[pp*self.train_pose_framelen:(pp+1)*self.train_pose_framelen])
        # #     with torch.no_grad():
        # #         z_con_batch, _  = self.con_ae(torch.stack(con_batch)[:, None, :, :])
        # #         _, z_emo_batch, _ = self.emo_ae(torch.stack(emo_batch)[:, None, :, :])
        # #     z_con_batch = z_con_batch.squeeze()
        # #     z_emo_batch = z_emo_batch.squeeze()
        # # else:
        # #     audio_batch = audio_batch.to(self.device) if not audio_batch.is_cuda else audio_batch
        # #     with torch.no_grad():
        # #         z_base, _ = self.base_ae(audio_batch)
        # #         z_base = z_base.squeeze()
        # #     z_con_batch, z_emo_batch = z_base, None
        
        motion_takes = z_motion_batch.shape[0]
        z_con_batch = audio_con[:motion_takes]
        z_emo_batch = audio_emo[:motion_takes] if audio_emo is not None else None
        z_sty_batch = audio_sty[:motion_takes] if audio_sty is not None else None
        return {
            "z_motion": z_motion_batch,
            "z_con": z_con_batch,
            "z_emo": z_emo_batch,
            "z_sty": z_sty_batch,
        }
    
    def _prepare_upper_body_data(self, motion):
        motion = motion[:, :-3]
        motion = motion.reshape(motion.shape[0], -1, 55)
        motion = motion[:, :, [i for i in range(55) if i not in self.lower_body_jts]]
        return motion.reshape(motion.shape[0], -1)
    
    def _prepare_model(self, model, tgt_dict, verbose):
        model_dict = model.state_dict()
        updated_keys = 0
        for k, v in tgt_dict.items():
            if k in model_dict and v.size() == model_dict[k].size():
                model_dict[k] = v
                updated_keys += 1
            else:
                print("[LATDIFF] ====> Skipping loading model key: ", k)
                raise Exception("[LATDIFF] ====> Cannot load model, key mismatch: ", k)
        model.to(self.device) 
        model.load_state_dict(model_dict)
        for p in model.parameters(): p.requires_grad = False
        model.eval()
        if verbose: print("[LATDIFF] ====> Loaded ", updated_keys, " keys out of ", len(model_dict.keys()), " keys")
        return model    


def mapinfo2takes(info, trainer=False):
    if not trainer: info = info.split("_")[1]
    if "happy" in info: return happy_takes
    elif "sad" in info: return sad_takes
    elif "angry" in info: return angry_takes
    elif "contempt" in info: return contempt_takes
    elif "disgust" in info: return disgust_takes
    elif "surprise" in info: return surprise_takes
    elif "fear" in info: return fear_takes
    else: raise Exception("Unknown emotion: ", info) 