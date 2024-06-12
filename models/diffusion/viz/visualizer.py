
import io
import wget
import gdown
import pydub
import pickle
import random
import string
import tarfile
import librosa
import subprocess
import numpy as np
import torchaudio
import scipy.io.wavfile
from einops import rearrange
import torchaudio.transforms as T
from tqdm import tqdm
from pathlib import Path
from pydub import AudioSegment

from dm.utils.bvh_utils import *
from dm.utils.facial_utils import *
from dm.utils.ldm_evals import subject2genderbeta, subject2genderbeta_1

class Visualizer:
    def __init__(self, config, b_path, processed):
        
        self.config = config
        self.blender_resrc_path = b_path
        self.tag = self.config["TRAIN_PARAM"]["tag"]
        self.viz_type = self.config["TRAIN_PARAM"][self.tag]["viz_type"]
        self.processed = processed

    def get_visualizer(self):
        if self.viz_type  == "CaMN":
            return CaMNVisualizer(self.config, self.blender_resrc_path, self.processed)
        elif self.viz_type  == "smpl-x":
            return SMPLXVisualizer(self.config, self.blender_resrc_path, self.processed)
        elif self.viz_type  == "HumanGenV3":
            return HumanGenVisualizer(self.config, self.blender_resrc_path, self.processed)
        else:
            raise NotImplementedError(f"Visualizer {self.viz_type } not implemented.")
        
class CaMNVisualizer():
    def __init__(self, config, blender_resrc_path, processed):
        self.config = config
        self.blender_resrc_path = blender_resrc_path
        self.processed = processed
        self.processed_path = self.processed / "processed-all-modalities"
        self.dirname = self.processed.parents[1]
        tag = self.config["TRAIN_PARAM"]["tag"]
        if tag == "diffusion":
            self.diffusion_type = self.config["TRAIN_PARAM"]["diffusion"]["arch"]
            with open(str(Path(self.dirname, "configs/", self.diffusion_type + ".json"))) as f:
                self.diffusion_cfg = json.load(f)
        
        
        if tag in ["latent_diffusion"]: 
            self.bvh_fps = self.config["DATA_PARAM"]["Bvh"]["fps"]
            self.bvh_save_path = self.processed_path / self.config["TRAIN_PARAM"]["motionprior"]["all_processed_motion"]
            self.smpl_viz_mode = self.config["TRAIN_PARAM"][tag]["smpl_viz_mode"]
            self.subjects = ["wayne", "scott", "solomon", "lawrence", "stewart", \
                           "nidal", "zhao", "lu", "zhang", "carlos", \
                           "jorge", "itoi", "daiki", "jaime", "li", \
                           "carla", "sophie", "catherine", "miranda", "kieks", \
                           "ayana", "luqi", "hailing", "kexin", "goto", \
                           "reamey", "yingqing", "tiffnay", "hanieh", "katya"]
            self.half_body = self.config["TRAIN_PARAM"][tag]["half_body"]
            self.lock_pymo_root = self.config["TRAIN_PARAM"][tag]["lock_pymo_root"]
            self.smplx_rep = self.config["TRAIN_PARAM"]["latent_diffusion"]["smplx_rep"]
        else: raise NotImplementedError(f"[CaMNVisualizer] Tag {tag} not implemented.")
        if isinstance(self.bvh_save_path, str): self.bvh_save_path = Path(self.bvh_save_path)
        
        # clothed/ not-clothed (diffusion, motionprior, motionprior_long)
        if tag != "latent_diffusion":
            self.clothed_meshes = self.config["TRAIN_PARAM"]["motionprior"]["clothed"] if tag == "motionprior_long" else self.config["TRAIN_PARAM"][tag]["clothed"]
            self.blender_clothed_arg = "clothed" if self.clothed_meshes else "naked" 
        else: self.clothed_meshes = False     
        
        # face viz: https://arkit-face-blendshapes.com/
        
        backup_experiment = self.config["TRAIN_PARAM"]["backup_experiment"]
        if not backup_experiment:
            with open(str(Path(self.processed, "eng_data_processed/all_eng_extracted_data.pkl")), "rb") as f:
                self.all_data = pickle.load(f)
        else: 
            with open(str(Path(self.processed.parents[1], "configs", "backup_data.json")), "r+") as f: self.backup_cfg = json.load(f)
            with open(self.backup_cfg["all_eng_extracted_pkl"], "rb") as f:
                self.all_data = pickle.load(f)
    
    def render_baselines(self, bvh_file, wav, target_path, endFrame, bmap):
        
        target_path = Path(target_path)
        # motion
        if bmap == "beat2smpl":
            fbx_file = Path(bvh_file)
            # if not fbx_file.exists(): subprocess.call([self.blender_exe, "-b", "-P", str(self.blender_resrc_path / "retarget_smpl_CamnInfer.py"), "--", bvh_file, str(target_path), "camninfer2smpl"])
            # else: print(f"FBX file {fbx_file} already exists.")
        else:
            fbx_file = Path(bvh_file).with_suffix(".fbx")
            if not fbx_file.exists(): subprocess.call([self.blender_exe, "-b", "-P", str(self.blender_resrc_path / "retarget_smpl.py"), "--", bvh_file, str(target_path), bmap])
            else: print(f"FBX file {fbx_file} already exists.")
            newstem = "_".join(fbx_file.stem.split("_")[1:])
            fbx_file.rename(target_path / f"{newstem}.fbx")
            fbx_file = target_path / f"{newstem}.fbx"
        # render
        render = target_path / f"{bvh_file.stem}_base_render.mp4"
        render_fps = int(endFrame//wav.duration_seconds) if not bmap == "beat2smpl" else 15 # TODO: (15 for FBX) remove hardcoding
        subprocess.call([self.blender_exe, "-b", "-P", str(self.blender_resrc_path / "render_smpl_blversion.py"), "--", 
                         str(fbx_file), str(endFrame), str(render), self.smpl_viz_mode, str(render_fps), bmap, str(self.half_body)])
        # add audio
        with_audio = target_path / f"{bvh_file.stem}_waudio.mp4"
        audio_file = target_path / f"{bvh_file.stem}_audio.wav"
        wav.export(str(audio_file), format="wav")
        subprocess.call([
            "ffmpeg", "-i", str(render), "-i", str(audio_file), "-c:v", "copy", "-c:a", "aac", str(with_audio)
        ])
        # # add text
        # tgt_video = target_path / f"{bvh_file.stem}_fullGT.mp4" # TODO: change baseline name in first_line
        # first_line = f"XXX {bvh_file.stem.split('_')[0]} {bvh_file.stem.split('_')[1]} {bvh_file.stem.split('_')[-1]}"
        # txt = "drawtext=fontfile=/usr/share/fonts/truetype/freefont/FreeMono.ttf:text='" + first_line + "':fontcolor=black:fontsize=18:x=10:y=10"
        # _ = subprocess.call([
        # "ffmpeg", "-i", with_audio, "-vf" , txt, "-codec:a", "copy", str(tgt_video)
        # ])  
        generated_files = sorted(Path(target_path).glob("*"))
        generated_files = [str(f) for f in generated_files]
        for f in generated_files:  
            if not any([x in f for x in ["_waudio.mp4"]]): os.remove(f)

    def render_GT(self, tgt_npz, tgt_video, audio_cut, withtext, framerate, show_flag=False, moshv1=False):
        # base render
        load_npz = np.load(tgt_npz, allow_pickle=True)
        endFrame = load_npz["poses"].shape[0]
        base_render = tgt_video.parent / f"{tgt_video.stem}_base_render.mp4"
        render_script = "render_smpl_half.py" if self.half_body else "render_smpl.py" # only change: camera pos and wall pos TODO merge both scripts
        if moshv1: render_script = "render_smpl_half_copy.py"
        if show_flag: render_script = "render_smpl_show.py"
        subprocess.call([self.blender_exe, "-b", "-P", str(self.blender_resrc_path / render_script), "--", 
                        str(tgt_npz), str(endFrame), str(base_render), self.smpl_viz_mode, str(framerate)])
        if withtext:
            # add audio
            with_audio = tgt_video.parent / f"{tgt_video.stem}_waudio.mp4"
            audio_file = tgt_video.parent / f"{tgt_video.stem}_audio.wav"
            audio_cut.export(str(audio_file), format="wav")
            _ = subprocess.call([
                "ffmpeg", "-i", str(base_render), "-i", str(audio_file), "-c:v", "copy", "-c:a", "aac", str(with_audio)
            ])
            # add text
            first_line = f"GT {tgt_video.stem.split('_')[0]} {tgt_video.stem.split('_')[1]} {tgt_video.stem.split('_')[-1]}"
            txt = "drawtext=fontfile=/usr/share/fonts/truetype/freefont/FreeMono.ttf:text='" + first_line + "':fontcolor=black:fontsize=18:x=10:y=10"
            _ = subprocess.call([
            "ffmpeg", "-i", with_audio, "-vf" , txt, "-codec:a", "copy", str(tgt_video)
            ])
        else:
            # add audio only
            audio_file = tgt_video.parent / f"{tgt_video.stem}_audio.wav"
            audio_cut.export(str(audio_file), format="wav")
            _ = subprocess.call([
                "ffmpeg", "-i", str(base_render), "-i", str(audio_file), "-c:v", "copy", "-c:a", "aac", str(tgt_video)
            ])
        generated_files = sorted(Path(tgt_video).parent.glob("*"))
        generated_files = [str(f) for f in generated_files]
        for f in generated_files:  
            if not any([x in f for x in ["_fullGT.mp4", "_GT.mp4"]]): os.remove(f)
    
    def animate_ldm_sample_v2(self, sample_dict, video_path, smplx_data=True, skip_trans=False, without_txt=False):
        
        assert smplx_data, "Only SMPL-X data supported for now in animate_ldm_sample_v2."
        feats = sample_dict["feats"] 
        audio = sample_dict["audio"]
        attr = sample_dict["info"]
        subject = [x for x in self.subjects if x in attr][0]
        samples = feats.shape[0]
        video_path_r = video_path
        smplx_TPOSE = self.blender_resrc_path / "resources" / "SMPLX_TPOSE_FLAT.bvh"
        self.blender36_exe = self.config["TRAIN_PARAM"]["diffusion"]["blender"]["cluster_36"] # blender 3.6 with modified io_bvh addon
        
        for i, feat in zip(range(samples), feats):
            # paths, filenames
            video_path = video_path_r / f"seq_{i}"
            video_path.mkdir(parents=True, exist_ok=True)
            x = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(6))
            self.ldm_file = f"seq_{i}_{x}"
            # audio
            current_audio = audio[i*10000:(i+1)*10000]
            audio_path = str(video_path / f"{self.ldm_file}_audio.wav")
            current_audio.export(audio_path, format="wav")
            # motion
            feat = feat.clone().detach().cpu().numpy()
            bvh_file_name = f"{subject}_{self.ldm_file}_motion"
            
            low_body_idx = [1, 2, 4, 5, 7, 8, 10, 11] # Lock below hips
            feat = rearrange(feat, "b (j d) -> b j d", d=3)
            assert feat.shape[1] == 56, f"SMPL-X data should have 56 joints, got {feat.shape[1]}."
            feat, trans = feat[:, :-1, :], feat[:, -1, :]
            # zeroing jaw (TODO why jaw is not zero?)
            feat = np.concatenate((feat[:, :22, :], np.zeros((feat.shape[0], 1, 3)), feat[:, 23:, :]), axis=1) 
            
            # 3 manipulation options
            if self.config["TRAIN_PARAM"]["latent_diffusion"]["zero_trans"]: 
                trans = np.zeros((feat.shape[0], 3))
                if self.config["TRAIN_PARAM"]["latent_diffusion"]["freeze_init_LoBody"]: 
                    low_body_idx = [1, 2, 4, 5, 7, 8, 10, 11]
                    init_low_body_pose = feat[0, low_body_idx, :]
                    feat[:, low_body_idx, :] = init_low_body_pose
                render_script = "render_smpl.py" 
            elif self.config["TRAIN_PARAM"]["latent_diffusion"]["half_body"]: 
                trans = np.zeros((feat.shape[0], 3))
                low_body_idx = [1, 2, 4, 5, 7, 8, 10, 11]
                init_low_body_pose = feat[0, low_body_idx, :]
                feat[:, low_body_idx, :] = init_low_body_pose
                render_script = "render_smpl_half.py"
            else: render_script = "render_smpl_1.py"

            smplx_npz = os.path.join(video_path, f"{subject}_{self.ldm_file}_motion_smplx.npz")
            mocap_frame_rate_ = np.array(self.bvh_fps, dtype='float64')
            # g, b = subject2genderbeta_1(subject) # MOSHED V1 shapes -- dont use updated shapes since rest of the project is based on old Mosh
            g, b = subject2genderbeta(subject)
            np.savez(
                smplx_npz,
                poses=feat,
                trans=trans,
                gender=g, betas=b,
                mocap_frame_rate=mocap_frame_rate_
            )
                
            # render
            render = video_path / f"{self.ldm_file}_render_video.mp4"
            subprocess.call([self.blender_exe, "-b", "-P", str(self.blender_resrc_path / render_script), "--", 
                                str(smplx_npz), str(feat.shape[0]), str(render), self.smpl_viz_mode, str(self.bvh_fps)])
            # add audio
            with_audio = video_path / f"{self.ldm_file}_waudio_video.mp4"
            _ = subprocess.call([
                "ffmpeg", "-i", str(render), "-i", str(audio_path), "-c:v", "copy", "-c:a", "aac", str(with_audio)
            ])
            # overlay text
            final_vid = video_path / f"{self.ldm_file}_single_subject_video.mp4"
            first_line = sample_dict["info"]
            if "swap_info" in sample_dict:
                second_line = sample_dict["swap_info"]
                txt = "drawtext=fontfile=/usr/share/fonts/truetype/freefont/FreeMono.ttf:text='" + first_line + "':fontcolor=black:fontsize=18:x=10:y=10,drawtext=fontfile=/usr/share/fonts/truetype/freefont/FreeMono.ttf:text='" + second_line + "':fontcolor=black:fontsize=18:x=10:y=30"
            else: txt = "drawtext=fontfile=/usr/share/fonts/truetype/freefont/FreeMono.ttf:text='" + first_line + "':fontcolor=black:fontsize=18:x=10:y=10"
            _ = subprocess.call([
            "ffmpeg", "-i", with_audio, "-vf" , txt, "-codec:a", "copy", str(final_vid)
            ])
            # delete extra files
            generated_files = sorted(video_path.glob("*"))
            generated_files = [str(f) for f in generated_files]
            npz_list = [bvh_file_name, ".npz"]
            if without_txt: 
                nontxt_vid_list = ["_waudio_video.mp4"]
                for f in generated_files:  
                    # if f != str(final_vid) and not all(x in f for x in npz_list) and not any(x in f for x in nontxt_vid_list): os.remove(f)
                    if not all(x in f for x in npz_list) and not any(x in f for x in nontxt_vid_list): os.remove(f) # removing the overlay text video
            else:
                for f in generated_files:  
                    if f != str(final_vid) and not all(x in f for x in npz_list): os.remove(f)

    def render_bvh2smplxbvh2npz(self, bvh_file, target_path, wav):
        smplx_TPOSE = self.blender_resrc_path / "resources" / "SMPLX_TPOSE_FLAT.bvh"
        self.blender36_exe = self.config["TRAIN_PARAM"]["diffusion"]["blender"]["cluster_36"] # blender 3.6 with modified io_bvh addon
        subprocess.call([self.blender36_exe, "-b", "-P", str(self.blender_resrc_path / "retarget_smpl2bvh2beatnpzWbetas.py"), "--", str(bvh_file), str(target_path), str(smplx_TPOSE), "infer_pipe"])
        smplx_npz = Path(target_path) / f"{Path(bvh_file).stem}.npz"
        load_npz = np.load(smplx_npz, allow_pickle=True)
        poses, trans = load_npz["poses"], load_npz["trans"]
        trans = np.zeros((poses.shape[0], 3))
        low_body_idx = [1, 2, 4, 5, 7, 8, 10, 11]
        init_low_body_pose = poses[0, low_body_idx, :]
        poses[:, low_body_idx, :] = init_low_body_pose
        render_script = "render_smpl_half.py"
        smplx_npz = Path(target_path) / f"{smplx_npz.stem}_adjusted.npz"
        mocap_frame_rate_ = np.array(self.bvh_fps, dtype='float64')
        subject = bvh_file.stem.split("_")[0]
        g, b = subject2genderbeta(subject)
        np.savez(
            smplx_npz,
            poses=poses,
            trans=trans,
            gender=g, betas=b,
            mocap_frame_rate=mocap_frame_rate_
        )
        # render
        target_path = Path(target_path) if isinstance(target_path, str) else target_path
        render = target_path / f"{smplx_npz.stem}_render_video.mp4"
        subprocess.call([self.blender_exe, "-b", "-P", str(self.blender_resrc_path / render_script), "--", 
                            str(smplx_npz), str(poses.shape[0]), str(render), self.smpl_viz_mode, str(self.bvh_fps)])
        # add audio
        with_audio = target_path / f"{smplx_npz.stem}_waudio_video.mp4"
        _ = subprocess.call([
            "ffmpeg", "-i", str(render), "-i", str(wav), "-c:v", "copy", "-c:a", "aac", str(with_audio)
        ])
        generated_files = sorted(target_path.glob("*"))
        generated_files = [str(f) for f in generated_files]
        endswith = "_waudio_video.mp4"
        for f in generated_files:
            if not str(f).endswith(endswith): os.remove(f)
        
    def animate_ldm_sample_v1(self, sample_dict, video_path, smplx_data, skip_trans=False, without_txt=False):
        
        feats = sample_dict["feats"] 
        audio = sample_dict["audio"]
        attr = sample_dict["info"]
        subject = [x for x in self.subjects if x in attr][0]
        samples = feats.shape[0]
        video_path_r = video_path
        smplx_TPOSE = self.blender_resrc_path / "resources" / "SMPLX_TPOSE_FLAT.bvh"
        self.blender36_exe = self.config["TRAIN_PARAM"]["diffusion"]["blender"]["cluster_36"] # blender 3.6 with modified io_bvh addon
        
        for i, feat in zip(range(samples), feats):
            # paths, filenames
            video_path = video_path_r / f"seq_{i}"
            video_path.mkdir(parents=True, exist_ok=True)
            x = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(6))
            self.ldm_file = f"seq_{i}_{x}"
            # audio
            current_audio = audio[i*10000:(i+1)*10000]
            audio_path = str(video_path / f"{self.ldm_file}_audio.wav")
            current_audio.export(audio_path, format="wav")
            # motion
            feat = feat.clone().detach().cpu().numpy()
            bvh_file_name = f"{subject}_{self.ldm_file}_motion"
            if not smplx_data:
                bvh_file = pymo_inverse_pipeline(feat, self.bvh_save_path, video_path, bvh_file_name, self.bvh_fps, verbose=False, viz_only=True)
                subprocess.call([self.blender36_exe, "-b", "-P", str(self.blender_resrc_path / "retarget_smpl2bvh.py"), "--", str(bvh_file), str(video_path), str(smplx_TPOSE), "infer_pipe"])
                smplx_npz = str(Path(bvh_file).with_suffix(".npz"))
                
                if self.lock_pymo_root:
                    load_npz = np.load(smplx_npz, allow_pickle=True)
                    poses, trans = load_npz["poses"], load_npz["trans"]
                    low_body_idx = [1, 2, 4, 5, 7, 8, 10, 11]
                    init_low_body_pose = poses[0, low_body_idx, :]
                    poses[:, low_body_idx, :] = init_low_body_pose
                    trans = trans[0].reshape(-1, 3) * np.ones((trans.shape[0], 1))
                    np.savez(
                        smplx_npz,
                        poses=poses,
                        trans=trans,
                        gender=load_npz["gender"], betas=load_npz["betas"],
                        mocap_frame_rate=np.array(self.bvh_fps, dtype='float64')
                    )
            
            else:
                
                low_body_idx = [1, 2, 4, 5, 7, 8, 10, 11] # Lock below hips
                feat = rearrange(feat, "b (j d) -> b j d", d=3)
                if feat.shape[1] == 56: feat = feat[:, :-1, :]
                
                if feat.shape[1] == 55:
                    init_low_body_pose = feat[0, low_body_idx, :]
                    feat[:, low_body_idx, :] = init_low_body_pose
                elif feat.shape[1] == 47:
                    feat = np.insert(feat, low_body_idx, np.zeros((feat.shape[0], 8, 3)), axis=1)
                
                trans = np.zeros((feat.shape[0], 3))
                smplx_npz = os.path.join(video_path, f"{subject}_{self.ldm_file}_motion_smplx.npz")
                mocap_frame_rate_ = np.array(self.bvh_fps, dtype='float64')
                g, b = subject2genderbeta(subject)
                np.savez(
                    smplx_npz,
                    poses=feat,
                    trans=trans,
                    gender=g, betas=b,
                    mocap_frame_rate=mocap_frame_rate_
                )
                
            # render
            render = video_path / f"{self.ldm_file}_render_video.mp4"
            render_script = "render_smpl_half.py" if self.half_body else "render_smpl.py" # only change: camera pos and wall pos TODO merge both scripts
            subprocess.call([self.blender_exe, "-b", "-P", str(self.blender_resrc_path / render_script), "--", 
                                str(smplx_npz), str(feat.shape[0]), str(render), self.smpl_viz_mode, str(self.bvh_fps)])
            # add audio
            with_audio = video_path / f"{self.ldm_file}_waudio_video.mp4"
            _ = subprocess.call([
                "ffmpeg", "-i", str(render), "-i", str(audio_path), "-c:v", "copy", "-c:a", "aac", str(with_audio)
            ])
            # overlay text
            final_vid = video_path / f"{self.ldm_file}_single_subject_video.mp4"
            first_line = sample_dict["info"]
            if "swap_info" in sample_dict:
                second_line = sample_dict["swap_info"]
                txt = "drawtext=fontfile=/usr/share/fonts/truetype/freefont/FreeMono.ttf:text='" + first_line + "':fontcolor=black:fontsize=18:x=10:y=10,drawtext=fontfile=/usr/share/fonts/truetype/freefont/FreeMono.ttf:text='" + second_line + "':fontcolor=black:fontsize=18:x=10:y=30"
            else: txt = "drawtext=fontfile=/usr/share/fonts/truetype/freefont/FreeMono.ttf:text='" + first_line + "':fontcolor=black:fontsize=18:x=10:y=10"
            _ = subprocess.call([
            "ffmpeg", "-i", with_audio, "-vf" , txt, "-codec:a", "copy", str(final_vid)
            ])
            # delete extra files
            generated_files = sorted(video_path.glob("*"))
            generated_files = [str(f) for f in generated_files]
            npz_list = [bvh_file_name, ".npz"]
            if without_txt: 
                nontxt_vid_list = ["_waudio_video.mp4"]
                for f in generated_files:  
                    if f != str(final_vid) and not all(x in f for x in npz_list) and not any(x in f for x in nontxt_vid_list): os.remove(f)
            else:
                for f in generated_files:  
                    if f != str(final_vid) and not all(x in f for x in npz_list): os.remove(f)

    def load_in_blender(self, EXEC_ON_CLUSTER):
        # NOTE: Use blender conda environment instead of default blender environment
        
        self.blender_exe = self.config["TRAIN_PARAM"]["diffusion"]["blender"]["local"] if not EXEC_ON_CLUSTER else self.config["TRAIN_PARAM"]["diffusion"]["blender"]["cluster"]
        self.blender_cfg_path = self.config["TRAIN_PARAM"]["diffusion"]["blender"]["local_config"] if not EXEC_ON_CLUSTER else self.config["TRAIN_PARAM"]["diffusion"]["blender"]["cluster_config"]
        # blender console in terminal: /snap/bin/blender -b --python-console    # obsolete
        
        # Plugins:
        # 1. Stop-motion-OBJ                                                    # legacy
        # 2. Face Baker                                                         # custom
        # 3. MakeHuman eXchange                                                 # modified
        # 4. Retarget BVH                                                       # modified
        
        # 1 download Stop-motion-OBJ addon
        smo_addon_file = self.blender_resrc_path / "addons/Stop-motion-OBJ-v2.1.1.zip"
        if not smo_addon_file.is_file(): 
            print(f"[BLENDER] Downloading Stop-motion-OBJ addon...")
            smo_addon_dwnld = "https://github.com/neverhood311/Stop-motion-OBJ/releases/download/v2.1.1/Stop-motion-OBJ-v2.1.1.zip"
            wget.download(smo_addon_dwnld, out = str(self.blender_resrc_path / "addons/")  )
            # using Gdrive file
            # addon_dwnld = "https://drive.google.com/file/d/1h-9gwaGR6dTQI33Cnqv7BOr7_VEjvjYF/view?usp=sharing"
            # gdown.download(addon_dwnld, "addons/Stop-motion-OBJ-v2.1.1.zip", quiet=False, fuzzy=True)
        
        # 2. Face Baker addon
        # Nothing to do here, script part of repo
        
        # 3. MakeHuman eXchange addon
        # Additionally, 9_export_mhx2 for MakeHuman export: https://drive.google.com/file/d/1EQgP5tucws6iqAflzh1VRkcaXF4z6CMC/view?usp=sharing
        mhx2_addon_file = self.blender_resrc_path / "addons/import_runtime_mhx2.tar.xz"
        if not mhx2_addon_file.is_file(): 
            print(f"[BLENDER] Downloading Import MHX2 addon...")
            mhx2_addon_dwnld = "https://drive.google.com/file/d/1vN_lh-CEBZ9jbg2EGShiFI3h6RcmUIIm/view?usp=sharing"
            gdown.download(mhx2_addon_dwnld, str(mhx2_addon_file), quiet=False, fuzzy=True)
            
        # 4. retarget bvh addon
        rtgt_addon_file = self.blender_resrc_path / "addons/retarget_bvh.tar.xz"
        if not rtgt_addon_file.is_file(): 
            print(f"[BLENDER] Downloading Import MHX2 addon...")
            rtgt_addon_dwnld = "https://drive.google.com/file/d/1m7LHjK1SAceVwY3eowJswUbIhHbuR9HP/view?usp=share_link"
            gdown.download(rtgt_addon_dwnld, str(rtgt_addon_file), quiet=False, fuzzy=True)
        
        # Resources:
        # 1. Face fbx                                                           # modified from https://arkit-face-blendshapes.com/ 
        # 2. Male, female body MHX2                                             # custom
        
        # 1. Face fbx
        resources_path = self.blender_resrc_path / "resources"
        resources_path.mkdir(parents=True, exist_ok=True)
        self.face_fbx_file = resources_path / "face_neutral.fbx"
        if not self.face_fbx_file.is_file(): 
            print(f"[BLENDER] Downloading face neutral mesh...")
            face_fbx_dwnld = "https://drive.google.com/file/d/1NPEP0P4w7EKNLry6TRapB-b71ulpvvBQ/view?usp=sharing"
            gdown.download(face_fbx_dwnld, str(self.face_fbx_file), quiet=False, fuzzy=True)
        
        # 2.a. male, female body meshes
        body_mesh_file = resources_path / "body_meshes.tar.xz"
        if not body_mesh_file.is_file(): 
            print(f"[BLENDER] Downloading body meshes...")
            body_mesh_dwnld = "https://drive.google.com/file/d/1wdeD4kCqP78iGIF4zsC3plKIHdFHqWi1/view?usp=sharing"
            gdown.download(body_mesh_dwnld, str(body_mesh_file), quiet=False, fuzzy=True)
        self.mesh_male = resources_path / "body_meshes/new_mesh/base_male.mhx2"
        self.mesh_female = resources_path / "body_meshes/new_mesh/base_female.mhx2"
        
        # 2.b. male, female clothed body meshes
        body_clothed_mesh_file = resources_path / "clothed_mesh.tar.xz"
        if not body_clothed_mesh_file.is_file(): 
            print(f"[BLENDER] Downloading body meshes...")
            body_clothed_mesh_dwnld = "https://drive.google.com/file/d/1fN5TfZFy8qBd3gFF0JonCQeAtl37iHxI/view?usp=share_link"
            gdown.download(body_clothed_mesh_dwnld, str(body_clothed_mesh_file), quiet=False, fuzzy=True)
        self.mesh_clothed_male = resources_path / "clothed_mesh/clothed_mesh/base_male.mhx2"
        self.mesh_clothed_female = resources_path / "clothed_mesh/clothed_mesh/base_female.mhx2"
        
        # Downloads complete!
        
        # Loading:
        # 1. extraction of MakeHuman eXchange and retarget bvh addons
        # 2. Stop-motion-OBJ and facebaker through enableaddon.py
        # 3. face fbx
        # 4. body meshes with IK
        # 5. select body mesh
        
        # 1. extraction of MakeHuman eXchange and retarget bvh addons
        def mhx2_members(tf):
            l = len("import_runtime_mhx2/")
            for member in tf.getmembers():
                if member.path.startswith("import_runtime_mhx2/"):
                    member.path = member.path[l:]
                    yield member
        def rtgt_members(tf):
            l = len("retarget_bvh/")
            for member in tf.getmembers():
                if member.path.startswith("retarget_bvh/"):
                    member.path = member.path[l:]
                    yield member        
        self.blender_installation = Path(self.blender_exe).parent
        mhx2_addon_folder = self.blender_installation / "3.4/scripts/addons/import_runtime_mhx2"
        mhx2_addon_folder.mkdir(parents=True, exist_ok=True)
        if self._directory_is_empty(mhx2_addon_folder):
            print("[BLENDER] Extracting MakeHuman eXchange addon...")
            with tarfile.open(str(mhx2_addon_file)) as tar:
                tar.extractall(str(mhx2_addon_folder), members=mhx2_members(tar))
            assert not self._directory_is_empty(mhx2_addon_folder), "[BLENDER] MakeHuman eXchange addon extraction failed!"
        rtgt_addon_folder = self.blender_installation / "3.4/scripts/addons/retarget_bvh"  
        rtgt_addon_folder.mkdir(parents=True, exist_ok=True)
        if self._directory_is_empty(rtgt_addon_folder):
            print("[BLENDER] Extracting Retarget BVH addon...")
            with tarfile.open(str(rtgt_addon_file)) as tar:
                tar.extractall(str(rtgt_addon_folder), members=rtgt_members(tar))
            assert not self._directory_is_empty(rtgt_addon_folder), "[BLENDER] Retarget BVH addon extraction failed!"          
                
        # 2. Stop-motion-OBJ and facebaker through enableaddon.py + enable all addons, save user preferences + auto run python script
        # Following call only for verification, use interactive blender console to enable addon  
        # executed in end of function to include smpl related addons
        # _ = subprocess.run([self.blender_exe, "-b", "-P", str(self.blender_resrc_path / "enableaddon.py") , "--", str(self.blender_resrc_path)])        
        
        # 3. face fbx
        # Nothing to do here
        
        # 4. body meshes with IK
        body_mesh_folder = self.blender_resrc_path / "resources/body_meshes"
        body_mesh_folder.mkdir(parents=True, exist_ok=True)
        with tarfile.open(str(body_mesh_file)) as tar: 
            tar.extractall(body_mesh_folder)
    
        body_clothed_mesh_folder = self.blender_resrc_path / "resources/clothed_mesh"
        body_clothed_mesh_folder.mkdir(parents=True, exist_ok=True)
        with tarfile.open(str(body_clothed_mesh_file)) as tar: 
            tar.extractall(body_clothed_mesh_folder)
    
        # 5. select body mesh
        if self.clothed_meshes:
            self.mesh_male = self.mesh_clothed_male
            self.mesh_female = self.mesh_clothed_female            
    
        # smpl specific loadings                                           
        proxy_picker = "Purchase from https://blendermarket.com/products/auto-rig-pro"
        beat2smpl_bmap = "https://drive.google.com/file/d/1G9qPe_3yOZMektokDZkDNoZm6jutFAzA/view?usp=share_link" # already part of modified ARP
        moglow2smplxbmap = "https://drive.google.com/file/d/1xOHHioDiwbK_iMkT7OMC0AQMvdi69U6F/view?usp=sharing"
        camninferbmap = "https://drive.google.com/file/d/13GZ5opQZf29DgJicwS5UpNg9nD4cWPn4/view?usp=sharing"
        ARP = "Purchase from https://blendermarket.com/products/auto-rig-pro"
        smplx =  "https://drive.google.com/file/d/1hdJ0GCqfJXbPWV0Vj236tgxu0IBVaxi3/view?usp=sharing"
    
        self.blender_cfg_path = Path(self.blender_cfg_path)
        proxy_picker_path = self.blender_cfg_path / "proxy_picker.py"
        if not proxy_picker_path.is_file(): 
            print(f"[BLENDER] Downloading proxy_picker addon...")
            gdown.download(proxy_picker, str(proxy_picker_path), quiet=False, fuzzy=True)
        
        ARP_path = self.blender_cfg_path / "auto_rig_pro-master.tar.xz"
        ARP_folder = self.blender_cfg_path / "auto_rig_pro-master"
        ARP_folder.mkdir(parents=True, exist_ok=True)
        if self._directory_is_empty(ARP_folder):
            print(f"[BLENDER] Downloading ARP addon...")
            gdown.download(ARP, str(ARP_path), quiet=False, fuzzy=True)
            print("[BLENDER] Extracting ARP addon...")
            with tarfile.open(str(ARP_path)) as tar:
                tar.extractall(str(self.blender_cfg_path))
            assert not self._directory_is_empty(ARP_folder), "[BLENDER] ARP addon extraction failed!"   
            if ARP_path.is_file(): ARP_path.unlink()    
        
        smplx_path = self.blender_cfg_path / "smplx_blender_addon.tar.xz"
        smplx_folder = self.blender_cfg_path / "smplx_blender_addon"
        smplx_folder.mkdir(parents=True, exist_ok=True)
        if self._directory_is_empty(smplx_folder): 
            print(f"[BLENDER] Downloading smplx addon...")
            gdown.download(smplx, str(smplx_path), quiet=False, fuzzy=True)
            print("[BLENDER] Extracting smplx addon...")
            with tarfile.open(str(smplx_path)) as tar:
                tar.extractall(str(self.blender_cfg_path))
            assert not self._directory_is_empty(smplx_folder), "[BLENDER] smplx addon extraction failed!"
            if smplx_path.is_file(): smplx_path.unlink()
        
        moglow2smplxbmap_file = self.blender_cfg_path / "auto_rig_pro-master/remap_presets/moglow2smplx.bmap"
        if not moglow2smplxbmap_file.is_file():
            print(f"[BLENDER] Downloading moglow2smplx.bmap...")
            gdown.download(moglow2smplxbmap, str(moglow2smplxbmap_file), quiet=False, fuzzy=True)
            
        camninferbmap_file = self.blender_cfg_path / "auto_rig_pro-master/remap_presets/camninfer2smpl.bmap"
        if not camninferbmap_file.is_file():
            print(f"[BLENDER] Downloading camninfer2smpl.bmap...")
            gdown.download(camninferbmap, str(camninferbmap_file), quiet=False, fuzzy=True)
        
        # Enable all addons
        _ = subprocess.run([self.blender_exe, "-b", "-P", str(self.blender_resrc_path / "enableaddon.py") , "--", str(self.blender_resrc_path)])   
            
        print("[BLENDER] setup complete!")

    def _directory_is_empty(self, directory: str) -> bool:
        return not any(directory.iterdir())    
        
  
        
class SMPLXVisualizer(CaMNVisualizer):
    def __init__(self, config, blender_resrc_path):
        self.config = config
        self.blender_resrc_path = blender_resrc_path
        super().__init__(self.config, self.blender_resrc_path)
        
    def pose(self, poses):
        pass
    
    def face(self, faces):
        pass
    
class HumanGenVisualizer(CaMNVisualizer):
    def __init__(self, config, blender_resrc_path):
        # TODO
        self.config = config
        self.blender_resrc_path = blender_resrc_path
        super().__init__(self.config, self.blender_resrc_path)
    
    def pose(self, poses):
        pass
    
    def face(self, faces):
        pass

# sample_bvh = "/home/kchhatre/Work/code/disentangled-s2g/viz_dump/prior_/demo_arp_smpl/wayne_0_1_1.bvh"   

# if __name__ == "__main__":
    
#     import json
#     with open("/home/kchhatre/Work/code/disentangled-s2g/configs/base.json", "r+") as f:
#         config = json.load(f)
#     EXEC_ON_CLUSTER = False
#     blender_resrc_path = Path("/home/kchhatre/Work/code/disentangled-s2g/models/diffusion/viz")
#     visualizer = CaMNVisualizer(config["TRAIN_PARAM"]["diffusion"], blender_resrc_path)
#     visualizer.load_in_blender(EXEC_ON_CLUSTER)
    
#     debug_recons = 0
    
#     visualizer.animate_batch(recons=debug_recons, src={}, viz_path = "", epoch = 0, debug=True)