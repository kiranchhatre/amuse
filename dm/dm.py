
import re
import pickle
import glob
import json
import time
import pydub
import math
import random
import datetime
import textgrid
import itertools
import pandas as pd
import lmdb 
import subprocess
import pyarrow as pa
from pathlib import Path
from tqdm import tqdm
from typing import List
from pydub import AudioSegment
from transformers import GPT2Tokenizer, GPT2Model, GPT2TokenizerFast

from .utils.bvh_utils import *
from .utils.wav_utils import *
from .utils.facial_utils import *
from .utils.corpus_utils import *
from .utils.all_words import *
from .utils.ldm_evals import *

from models import Pretrained_AST_EVP

class dm():
    
    def __init__(self, data_path, annotations_eng, processed, caches=None, prior_caches=None, latent_caches=None, config=None, backup_cfg=None):
        self.data_path = data_path
        self.processed = processed
        self.annotations_eng = annotations_eng
        self.config = config
        self.backup_cfg = backup_cfg
        self.tag = self.config["TRAIN_PARAM"]["tag"]
        self.emotions = {"netural": 0, "happiness": 1, "anger": 2, "sadness": 3,
                         "contempt": 4, "surprise": 5, "fear": 6, "disgust": 7}
        self.actor_attr = { 
            # 15 male/ 15 female, 10 Native/ 20 Non-native, 10 Caucasian/ 16 Asian/ 4 African
            "1" : ["wayne", "male", "US", "native", 25, "Caucasian"],
            "2" : ["scott", "male", "US", "native", 32, "Caucasian"],
            "3" : ["solomon", "male", "US", "native", 40, "African"],
            "4" : ["lawrence", "male", "Australia", "native", 26, "Asian"],
            "5" : ["stewart", "male", "UK", "native", 30, "Caucasian"],
            "6" : ["carla", "female", "US", "native", 27, "Caucasian"],
            "7" : ["sophie", "female", "US", "native", 30, "Caucasian"],
            "8" : ["catherine", "female", "US", "native", 31, "Asian"],
            "9" : ["miranda", "female", "UK", "native", 32, "Caucasian"],
            "10": ["kieks", "female", "UK", "native", 35, "Caucasian"],
            "11": ["nidal", "male", "Arab", "notnative", 38, "African"],
            "12": ["zhao", "male", "Thailand", "notnative", 32, "Asian"],
            "13": ["lu", "male", "China", "notnative", 25, "Asian"],
            "14": ["zhang", "male", "China", "notnative", 24, "Asian"],
            "15": ["carlos", "male", "China", "notnative", 40, "Asian"],
            "16": ["jorge", "male", "China", "notnative", 32, "Asian"],
            "17": ["itoi", "male", "Japan", "notnative", 32, "Asian"],
            "18": ["daiki", "male", "Japan", "notnative", 22, "Asian"],
            "19": ["jaime", "male", "Peru", "notnative", 27, "Caucasian"],
            "20": ["li", "male", "Spain", "notnative", 30, "Caucasian"],
            "21": ["ayana", "female", "China", "notnative", 31, "Asian"],
            "22": ["luqi", "female", "China", "notnative", 24, "Asian"],
            "23": ["hailing", "female", "China", "notnative", 26, "Asian"],
            "24": ["kexin", "female", "China", "notnative", 32, "Asian"],
            "25": ["goto", "female", "Japan", "notnative", 24, "Asian"],
            "26": ["reamey", "female", "Japan", "notnative", 26, "Asian"],
            "27": ["yingqing", "female", "Iran", "notnative", 31, "African"],
            "28": ["tiffnay", "female", "Jamaica", "notnative", 33, "African"],
            "29": ["hanieh", "female", "Jamaica", "notnative", 24, "Asian"],
            "30": ["katya", "female", "Russia", "notnative", 25, "Caucasian"]
        }
        
        # https://stackoverflow.com/questions/52232839/understanding-the-output-of-mfcc
        # Feature extraction viz: https://ar1st0crat.github.io/NWaves.Playground/extractors
        # 60 sec x 16000 Hz = 960000 samples / 128 = 7500 frames ~ 6751 x 13 = 87763 MFCC features with n_fft=2048 
        self.mfcc_transform = T.MFCC(sample_rate=config["DATA_PARAM"]["Wav"]["sample_rate"], 
                                n_mfcc=config["DATA_PARAM"]["Wav"]["n_mfcc"],                           # n_mfcc: Number of mfc coefficients to retain
                                melkwargs={"n_fft": config["DATA_PARAM"]["Wav"]["n_fft"],               # n_fft (also win_length): Size of FFT, creates n_fft // 2 + 1 bins. 
                                        "n_mels": config["DATA_PARAM"]["Wav"]["n_mels"],                # n_mels: Number of Mel filterbanks. (Default: 128)
                                        "hop_length": config["DATA_PARAM"]["Wav"]["hop_length"],        # hop_length: Length of hop between STFT windows. (Default: win_length // 2)
                                        "mel_scale": config["DATA_PARAM"]["Wav"]["mel_scale"],},)       # mel_scale: Scale to use: htk or slaney. (Default: htk)

        # LMDB CACHE 
        ########################################################################
        if self.tag == "latent_diffusion":
            self.train_pose_framelen = self.config["DATA_PARAM"]["Bvh"]["train_pose_framelen"]
            assert self.train_pose_framelen == 300, "[RAW PATHS/ LATDIFF] Latent diffusion implemented for 300 frame length"
            self.new_latdiff_cache, self.old_latdiff_cache = latent_caches[0], latent_caches[1]
            data_file = self.old_latdiff_cache / "data.mdb"
            if self.old_latdiff_cache.exists() and data_file.is_file():                         
                self.process_latdiff_cache = False
            else:
                self.process_latdiff_cache = True
            self.now = datetime.datetime.now().date()
            self.base_bvh_fps = self.config["DATA_PARAM"]["Bvh"]["fps"]
            self.use_pymo = self.config["DATA_PARAM"]["Bvh"]["pymo_based"]["use"]
            if self.use_pymo: 
                self.pymo_bvh_version = self.config["DATA_PARAM"]["Bvh"]["pymo_based"]["version"]
                self.fingers = self.config["DATA_PARAM"]["Bvh"]["pymo_based"]["fingers"]
            else: pass 
            self.smplx_data = self.config["TRAIN_PARAM"]["latent_diffusion"]["smplx_data"]
            if self.smplx_data:
                self.smplx_data_type = self.config["TRAIN_PARAM"]["latent_diffusion"]["smplx_data_type"] 
                if self.smplx_data_type == "ARP": smplx_data_id = "smplx_extract_path"
                elif self.smplx_data_type == "MOSH": 
                    BEAT_data_version = self.config["TRAIN_PARAM"]["wav_dtw_mfcc"]["ablation_version"]
                    if BEAT_data_version == "v0": smplx_data_id = "mosh_extract_path"
                    elif BEAT_data_version == "v1": smplx_data_id = "mosh_extract_path_v1"
                    else: raise ValueError(f"[RAW PATHS/ LATDIFF] Invalid BEAT data version: {BEAT_data_version}")
                self.smplx_extract_path = self.config["DATA_PARAM"]["Bvh"][smplx_data_id] 
            
            # # backup cfg
            # if self.backup_cfg is not None:
            #     print("[DM/ LATDIFF] Using backup cfg!")
            #     self.data_path = Path(self.backup_cfg["data_path"])
            #     self.process_latdiff_cache = False
                
            if not self.process_latdiff_cache: print("[RAW PATHS/ LATDIFF] Cache already exists. Skipping cache process.")
            else:
                print("[RAW PATHS/ LATDIFF] Cache will be created.")
                latdiff_cache_version = str(self.now)
                latdiff_cache_version += f"_{self.base_bvh_fps}F_"
                # latdiff_cache_version += f"{self.pymo_bvh_version}" 
                latdiff_cache_version += f"{self.pymo_bvh_version}" if self.use_pymo else "fing" 
                latdiff_cache_version += f"_smplx_{self.smplx_data_type}" if self.smplx_data else ""
                # AST models and dataset size versions
                if "ablation" in self.config["TRAIN_PARAM"]["wav_dtw_mfcc"]:
                    latdiff_cache_version += f"_{self.config['TRAIN_PARAM']['wav_dtw_mfcc']['ablation']}"
                    latdiff_cache_version += f"_{self.config['TRAIN_PARAM']['wav_dtw_mfcc']['ablation_version']}"
                    if self.config['TRAIN_PARAM']['wav_dtw_mfcc']['frame_based_feats']: latdiff_cache_version += f"_feat_based"
                latdiff_cache_version += f"_{self.train_pose_framelen}"
                Path(self.new_latdiff_cache).mkdir(parents=True, exist_ok=True)
                self.new_val_latdiff_cache = self.new_latdiff_cache / f"{latdiff_cache_version}_val" 
                self.new_latdiff_cache = self.new_latdiff_cache / latdiff_cache_version    
                print(f"[RAW PATHS/ LATDIFF] New latent diffusion cache path: {self.new_latdiff_cache}, with its val cache!")
                self.n_latdiff_out_samples = 0
                self.n_val_latdiff_out_samples = 0
                
        ########################################################################
        
        # Create all data pickles 
        self.all_data = {}
        all_eng_extracted_pkl = glob.glob(str(Path(self.processed, "eng_data_processed/all_eng_extracted_data.pkl")))
        if self.backup_cfg is None:                                             # backup cfg
            Path(self.processed, "eng_data_processed").mkdir(parents=True, exist_ok=True)
        else: all_eng_extracted_pkl = [self.backup_cfg["all_eng_extracted_pkl"]]
        if not all_eng_extracted_pkl:
            print("[RAW PATHS] Data not processed, creating all data path pickle...")
            
            # BVH
            if not glob.glob(str(Path(self.processed, "eng_data_processed/all_bvh.pkl"))):
                all_bvh = [str(f.resolve()) for f in self.data_path.rglob('*/*.bvh')]
                all_bvh.sort(key=str)
                with open(str(Path(self.processed, "eng_data_processed/all_bvh.pkl")), 'wb') as f:
                    pickle.dump(all_bvh, f)
                print("[RAW PATHS] (1/7) BVH pickle created.")
            else:
                with open(str(Path(self.processed, "eng_data_processed/all_bvh.pkl")), "rb") as f:
                    all_bvh = pickle.load(f)
                print("[RAW PATHS] (1/7) BVH pickle already exists, loaded.")
            
            # WAV       
            if not glob.glob(str(Path(self.processed, "eng_data_processed/all_wav.pkl"))):
                all_wav = [str(f.resolve()) for f in self.data_path.rglob('*/*.wav')]
                all_wav.sort(key=str)
                with open(str(Path(self.processed, "eng_data_processed/all_wav.pkl")), 'wb') as f:
                    pickle.dump(all_wav, f)
                print("[RAW PATHS] (2/7) WAV pickle created.")
            else:
                with open(str(Path(self.processed, "eng_data_processed/all_wav.pkl")), "rb") as f:
                    all_wav = pickle.load(f)
                print("[RAW PATHS] (2/7) WAV pickle already exists, loaded.")
            
            # TextGrid        
            if not glob.glob(str(Path(self.processed, "eng_data_processed/all_texts.pkl"))):
                all_texts = [str(f.resolve()) for f in self.data_path.rglob('*/*.TextGrid')]
                all_texts.sort(key=str)
                with open(str(Path(self.processed, "eng_data_processed/all_texts.pkl")), 'wb') as f:
                    pickle.dump(all_texts, f)
                print("[RAW PATHS] (3/7) TextGrid pickle created.")
            else:
                with open(str(Path(self.processed, "eng_data_processed/all_texts.pkl")), "rb") as f:
                    all_texts = pickle.load(f)
                print("[RAW PATHS] (3/7) TextGrid pickle already exists, loaded.")
            
            # JSON        
            if not glob.glob(str(Path(self.processed, "eng_data_processed/all_json.pkl"))):
                all_json = [str(f.resolve()) for f in self.data_path.rglob('*/*.json')]
                all_json.sort(key=str)
                with open(str(Path(self.processed, "eng_data_processed/all_json.pkl")), 'wb') as f:
                    pickle.dump(all_json, f)
                print("[RAW PATHS] (4/7) JSON pickle created.")
            else:
                with open(str(Path(self.processed, "eng_data_processed/all_json.pkl")), "rb") as f:
                    all_json = pickle.load(f)
                print("[RAW PATHS] (4/7) JSON pickle already exists, loaded.")
            
            # Emotions
            if not glob.glob(str(Path(self.processed, "eng_data_processed/all_emotions.pkl"))):
                all_emotions = [str(f.resolve()) for f in self.annotations_eng.rglob('*/*.csv')]
                all_emotions.sort(key=str)
                with open(str(Path(self.processed, "eng_data_processed/all_emotions.pkl")), 'wb') as f:
                    pickle.dump(all_emotions, f)
                print("[RAW PATHS] (5/7) Emotions pickle created.")
            else:
                with open(str(Path(self.processed, "eng_data_processed/all_emotions.pkl")), "rb") as f:
                    all_emotions = pickle.load(f)
                print("[RAW PATHS] (5/7) Emotions pickle already exists, loaded.")
            
            # Semantics
            if not glob.glob(str(Path(self.processed, "eng_data_processed/all_semantics.pkl"))):
                all_semantics = [str(f.resolve()) for f in self.annotations_eng.rglob('*/*.txt')]
                all_semantics.sort(key=str)
                with open(str(Path(self.processed, "eng_data_processed/all_semantics.pkl")), 'wb') as f:
                    pickle.dump(all_semantics, f)
                print("[RAW PATHS] (6/7) Semantics pickle created.")
            else:
                with open(str(Path(self.processed, "eng_data_processed/all_semantics.pkl")), "rb") as f:
                    all_semantics = pickle.load(f) 
                print("[RAW PATHS] (6/7) Semantics pickle already exists, loaded.")
            
            eng_noneng_bvhs = [i.split('/')[-1].split(".")[0] for i in all_bvh]
            noneng_bvhs = [i for i in eng_noneng_bvhs if i.split("_")[-3] not in ["0", "1"]] # Non-English BVHs
            eng_bvhs = set(eng_noneng_bvhs).difference(noneng_bvhs)
            
            eng_noneng_wavs = [i.split('/')[-1].split(".")[0] for i in all_wav]
            noneng_wavs = [i for i in eng_noneng_wavs if i.split("_")[-3] not in ["0", "1"]] # Non-English WAVs
            eng_wavs = set(eng_noneng_wavs).difference(noneng_wavs)
            
            eng_noneng_jsons = [i.split('/')[-1].split(".")[0] for i in all_json]
            noneng_jsons = [i for i in eng_noneng_jsons if i.split("_")[-3] not in ["0", "1"]] # Non-English JSONs
            eng_jsons = set(eng_noneng_jsons).difference(noneng_jsons)
            
            eng_noneng_texts = [i.split('/')[-1].split(".")[0] for i in all_texts]
            noneng_texts = [i for i in eng_noneng_texts if i.split("_")[-3] not in ["0", "1"]] # Non-English TextGrids  
            eng_texts = set(eng_noneng_texts).difference(noneng_texts)
            
            commons = eng_bvhs.intersection(eng_wavs).intersection(eng_jsons).intersection(eng_texts) 

            print("[RAW PATHS] Eng/ total bvhs: ", len(eng_bvhs), "/", len(eng_noneng_bvhs), 
                  ", Eng/ total wavs: ", len(eng_wavs), "/", len(eng_noneng_wavs), 
                  ", Eng/ total texts: ", len(eng_texts), "/", len(eng_noneng_texts), 
                  ", Eng/ total face jsons: ", len(eng_jsons), "/", len(eng_noneng_jsons),
                  ", all raw emotions: ", len(all_emotions), ", all raw semantics: ", len(all_semantics))
            print("[RAW PATHS] All modalities available for takes: ", len(commons), " in English only rawdata.")
            
            print("[RAW PATHS] Dropping takes wrt BVH for missing modalities: ")
            print("[RAW PATHS] Waves: ", eng_bvhs.difference(eng_wavs))
            print("[RAW PATHS] JSONs: ", eng_bvhs.difference(eng_jsons))
            print("[RAW PATHS] Texts: ", eng_bvhs.difference(eng_texts)) # {'7_sophie_1_3_3', '7_sophie_1_4_4', '7_sophie_1_9_9', '7_sophie_1_10_10'}
            print("[RAW PATHS] Emotions: ", eng_bvhs.difference(eng_texts)) # {'7_sophie_1_3_3', '7_sophie_1_4_4', '7_sophie_1_9_9', '7_sophie_1_10_10'}
            print("[RAW PATHS] Semantics: ", eng_bvhs.difference(eng_texts)) # {'7_sophie_1_3_3', '7_sophie_1_4_4', '7_sophie_1_9_9', '7_sophie_1_10_10'}

            # Full data extraction from BEAT-fulldata/BEAT/
            # Eng/ total bvhs:  530 /  697 
            # Eng/ total wavs:  527 /  685 
            # Eng/ total texts:  459 /  459 
            # Eng/ total face jsons:  527 /  685 
            # all emotions:  1945 
            # all semantics:  1945
            # All modalities available for takes:  457
            
            # English data extraction from beat-rawdata-eng/beat_rawdata_english/
            # Eng/ total bvhs:  1945 / 1945 
            # Eng/ total wavs:  1945 / 1945 
            # Eng/ total texts:  1941 / 1941 
            # Eng/ total face jsons:  1945 / 1945 
            # all emotions:  1945 
            # all semantics:  1945
            # All modalities available for takes:  1941
            
            for common in commons:
                actor = common.split("_")[1]
                take = "_".join(common.split("_")[2:])
                # from_emo, to_emo = common.split("_")[-2], common.split("_")[-1]
                if actor not in self.all_data:                
                    self.all_data[actor] = {}
                self.all_data[actor][take] = {}
                self.all_data[actor][take]["bvh"]  = [i for i in all_bvh if common in i]
                self.all_data[actor][take]["wav"]  = [i for i in all_wav if common in i]
                self.all_data[actor][take]["json"] = [i for i in all_json if common in i]
                self.all_data[actor][take]["txt"]  = [i for i in all_texts if common in i]
                self.all_data[actor][take]["emo"]  = [i for i in all_emotions if common in i]
                self.all_data[actor][take]["sem"]  = [i for i in all_semantics if common in i]
                # for i in range(int(from_emo), int(to_emo)+1):
                #     emo_id = "_".join(common.split("_")[:3])+str(i)+"_"+str(i)+".csv"
                #     sem_id = "_".join(common.split("_")[:3])+str(i)+"_"+str(i)+".txt"
                #     self.all_data[actor][take]["emo"].extend([j for j in all_emotions if emo_id in j])
                #     self.all_data[actor][take]["sem"].extend([k for k in all_semantics if sem_id in k])
            
            with open(str(Path(self.processed, "eng_data_processed/all_eng_extracted_data.pkl")), 'wb') as f:
                pickle.dump(self.all_data, f)
            print("[RAW PATHS] (7/7) All English data pickle created!")
            
        else:
            print("[RAW PATHS] All English data pickles already exist, loading...")
            with open(all_eng_extracted_pkl[0], "rb") as f:
                self.all_data = pickle.load(f)
            self.total_datapoints = sum(len(v) for v in self.all_data.values()) # 1941
            print("[RAW PATHS] All English pickle data loaded. Total datapoints: ", self.total_datapoints) 
            
            all_values = list(self._NestedDictValues(self.all_data))
            assert all(all_values) == True, "Modality paths are not available for all datapoints!"
    
    def emotionalpreprocess_v1(self, verbose=False, viz_inverse_pipe=False, preferred_viz=""):
        pass
            
    def preprocess(self, viz_inverse_pipe=False, verbose: bool = False):
        
        assert self.config["DATA_PARAM"]["Bvh"]["fps"] == self.config["DATA_PARAM"]["Json"]["fps"] == \
               self.config["DATA_PARAM"]["Txtgrid"]["fps"] == self.config["DATA_PARAM"]["Sem"]["fps"], \
               "FPS for all modalities should be same as BVH: {}, check config file.".format(self.config["DATA_PARAM"]["Bvh"]["fps"])
               
        # print("[PREPROCESS] Total datapoints: ", sum(len(v) for v in self.all_data.values())) # 1941
        all_common_takes = []
        for v in self.all_data.values():
            all_common_takes.append(list(v.keys()))
        all_common_takes = list(set(all_common_takes[0]).intersection(*all_common_takes)) # Drop uncommon content
        all_common_takes = [x for x in all_common_takes if x[0] != '1'] # Drop conversational data
        self.all_data = {k: {take: v[take] for take in all_common_takes} for k, v in self.all_data.items()}
        self.total_datapoints = sum(len(v) for v in self.all_data.values()) # 630
        print("[PREPROCESS] Total datapoints after dropping conversational data and uncommon content: ", self.total_datapoints) # 630

        print("[PREPROCESS] Preprocessing individual modalities...")        
        self.processed_path = Path(self.processed, "processed-all-modalities")
        self.processed_path.mkdir(parents=True, exist_ok=True)
        
        self.dtw_align_save_path = Path(self.processed_path, "aligned-dtw")
        self.dtw_align_save_path.mkdir(parents=True, exist_ok=True)
        if len(glob.glob(str(self.dtw_align_save_path) + '/*.pkl')) == self.total_datapoints: self.align_dtw = False
        else: self.align_dtw = True
            
        training_version = self.config["TRAIN_PARAM"][self.tag]["training_version"]
        if training_version == "v1":
            self.emotional_actors = ["1", "2", "3", "4", "5", "6", "7", "8", "9",              
                                     "10", "12", "13", "16", "18", "21", "26", "27", "30"]
            val_actors = ["11", "14", "15", "17", "19", "20"]
            self.emotional_actors.extend(val_actors)
            self.emotional_takes = ["65", "66", "73", "74", "81", "82", "87", "88", "95", "96", "103", "104", "111", "112"]
            neutral_takes: List[str] = ["9", "10"]
            self.emotional_takes.extend(neutral_takes)
        
        for i in tqdm(self.all_data.keys(), desc="[PREPROCESS] All actors", leave=False):
            for j in tqdm(self.all_data[i].keys(), desc="[PREPROCESS] Takes for an actor", leave=False):
                
                emo_file = self.all_data[i][j]["emo"][0]
                emo_file = Path(self.processed_path).parents[1] / emo_file.split("BEAT/")[1]
                emo = torch.from_numpy(np.genfromtxt(emo_file, delimiter=",")[-2:]) #  tensor([58.,  3.], dtype=torch.float64) as duration, emotion
                self.all_data[i][j]["emo_label"] = emo[-1].long()
                self.all_data[i][j]["emo_duration"] = emo[0].long()
                
        print(f"[PREPROCESS] Preprocessing finished!")
    
    def beat2smplnpz(self, extract, blender_resrc_path):    
        
        # SMPLX BVH + NPZ Conversion
        blender_exe = self.config["TRAIN_PARAM"]["diffusion"]["blender"]["cluster_36"]
        # NOTE: install ARP, smplx manually and move bmap preset in .config dir before execution
        
        pymo_engall_w_inconsistent = [str(f.resolve()) for f in self.data_path.rglob('*/*.bvh')] # 1945
        precomputed_inconsistents = ["29_hanieh_1_0_0", "29_hanieh_1_1_1",
                                    "29_hanieh_1_4_4", "29_hanieh_1_3_3",
                                    "19_jaime_1_1_1", "19_jaime_1_2_2",
                                    "26_reamey_1_2_2", "26_reamey_1_3_3",
                                    "26_reamey_1_1_1"]
        pymo_engall = [i for i in pymo_engall_w_inconsistent if Path(i).stem not in precomputed_inconsistents]
        pymo_engall.sort(key=str)
        
        smplx_bvh = blender_resrc_path / "resources" / "SMPLX_TPOSE_FLAT.bvh"
        if not smplx_bvh.is_file():
            print(f"[Beat2SMPLNPZ] SMPLX BVH not found at {smplx_bvh}, downloading...") 
            smplx_bvh_dwnld = "https://drive.google.com/file/d/1ss3JP4M66pAe86tqjA2okCsjyTqT5cs2/view?usp=sharing"
            import gdown
            gdown.download(smplx_bvh_dwnld, str(smplx_bvh), quiet=False, fuzzy=True)
        
        # bpy_36_resources = "REMOVED- following readme instructions for ARP and SMPLX"
        # Path(blender_cfg).mkdir(parents=True, exist_ok=True)
        # ARP_folder = Path(blender_cfg) / "auto_rig_pro-master"
        # smplx_folder = Path(blender_cfg) / "smplx_blender_addon"
        
        # if not all([ARP_folder.is_dir(), smplx_folder.is_dir()]):
        #     import gdown
        #     gdown.download(bpy_36_resources, str(blender_cfg), quiet=False, fuzzy=True)
        #     downloaded_file = next(Path(blender_cfg).rglob("*"))
        #     subprocess.call(["mv", str(downloaded_file), str(Path(blender_cfg) / "blender_resources.tar.xz")])
        #     subprocess.call(["tar", "-xf", str(Path(blender_cfg) / "blender_resources.tar.xz"), "-C", str(blender_cfg)])
        #     subprocess.call(["rm", str(Path(blender_cfg) / "blender_resources.tar.xz")])
        
        conversion_start = time.time()
        extract = Path(self.data_path).resolve().parents[2] / extract
        print("[Beat2SMPLFBX] Converting BVH to SMPLX NPZ...")
        for i, bvh_file in enumerate(pymo_engall):
            print(f"processing {i+1}/{len(pymo_engall)}: {bvh_file}")
            smplx_bvh_file = Path(extract) / f"{Path(bvh_file).stem}.bvh"
            smplx_npz = Path(extract) / f"{Path(bvh_file).stem}.npz"
            if not all([smplx_bvh_file.is_file(), smplx_npz.is_file()]):
                subprocess.call([blender_exe, "-b", "-P", str(blender_resrc_path / "retarget_smpl2bvh.py"), "--", str(bvh_file), extract, str(smplx_bvh)])
            assert all([smplx_bvh_file.is_file(), smplx_npz.is_file()]), f"SMPLX BVH and NPZ not found at {smplx_bvh_file} and {smplx_npz}"
            
            logfile = Path(extract) / f"{Path(bvh_file).stem}_smpl_blender.log"
            logfile.unlink(missing_ok=True)
        
        print(f"[Beat2SMPLNPZ] Conversion finished in {(time.time()-conversion_start)/60} minutes")

    def DTW_align_dm_ast(self):
        
        self.processed_path = Path(self.processed, "processed-all-modalities/fbanks")
        self.processed_path.mkdir(parents=True, exist_ok=True)
        disentagler_loader_file = Path(self.processed_path, "disentagler_loader.npz")
        
        if not disentagler_loader_file.is_file():
            
            print("[AST] Processing started...")
            target_length = self.config["TRAIN_PARAM"]["wav_dtw_mfcc"]["target_length"]
            for i in tqdm(self.all_data.keys(), desc="[AST] All actors", leave=False): 
                for j in tqdm(self.all_data[i].keys(), desc="[AST] Takes for an actor", leave=False):
                    e = self.all_data[i][j]["emo_label"].item()
                    wav_file = self.all_data[i][j]["wav"][0]
                    waveform, sr = torchaudio.load(wav_file)
                    assert sr == 16000, f"SR is {sr}, not 16000 for {wav_file}"
                    waveform = waveform - waveform.mean()
                    total_chunks = waveform.shape[1]//160000
                    for k in range(0, total_chunks):
                        sliced_chunk = waveform[:, k:k+160000]
                        fbank = torchaudio.compliance.kaldi.fbank(sliced_chunk, htk_compat=True, sample_frequency=sr, use_energy=False, window_type='hanning', 
                                                                num_mel_bins=self.config["TRAIN_PARAM"]["wav_dtw_mfcc"]["num_mel_bins"], dither=0.0, frame_shift=10)
                        if fbank.shape[0] > 300: # audio clip shorter than 300 frames are not considered
                            n_frames = fbank.shape[0]
                            p = target_length - n_frames
                            if p > 0:
                                m = torch.nn.ZeroPad2d((0, 0, 0, p))
                                fbank = m(fbank)
                            elif p < 0: fbank = fbank[0:target_length, :]
                            actor_id = int([k for k, v in self.actor_attr.items() if i in v][0])
                            self.all_data[i][j][f"ast_{k}"] = {"fbank": fbank, "emo_id": e, "actor_id": actor_id}
                            self.all_data[i][j]["chunks"] = total_chunks
                            
                            
            self.n_takes = ['0_9_9', '0_10_10']
            self.h_takes = ['0_65_65', '0_66_66']       # happy 65 - 72
            self.a_takes = ['0_73_73', '0_74_74']       # angry 73 - 80
            self.s_takes = ['0_81_81', '0_82_82']       # sad 81 - 86
            self.c_takes = ['0_87_87', '0_88_88']       # contempt 87 - 94
            self.su_takes = ['0_95_95', '0_96_96']      # surprise 95 - 102
            self.f_takes = ['0_103_103', '0_104_104']   # fear 103 - 110
            self.d_takes = ['0_111_111', '0_112_112']   # disgust 111 - 118
            self.emo_sorted_takes_1 = [self.n_takes, self.h_takes, self.a_takes, self.s_takes, self.c_takes, self.su_takes, self.f_takes, self.d_takes]
            
            disentagler_loader = {}
            all_actors = [v[0] for k, v in self.actor_attr.items()]
            v_actors = ["nidal", "li", "kexin"]
            t_actors = [x for x in all_actors if x not in v_actors and x not in ["yingqing", "goto"]] # yingqing and goto have incorrect emotion labels
            print(f"Training actors length: {len(t_actors)}, Validation actors length: {len(v_actors)}") # 25, 3
            
            lists = {"train": list(itertools.combinations(t_actors, 2)), "val": list(itertools.combinations(v_actors, 2))}
            for k, a_list in lists.items():
                processed_data = []
                for two_actors in tqdm(a_list, desc="[AST] All a-list combinations", leave=False): 
                    for emo_take in self.emo_sorted_takes_1:
                        takes_combo = list(itertools.combinations(emo_take, 2))
                        for take_combo in takes_combo:
                            ast_keys_a1_t1 = self.all_data[two_actors[0]][take_combo[0]]["chunks"]
                            ast_keys_a1_t2 = self.all_data[two_actors[0]][take_combo[1]]["chunks"]
                            ast_keys_a2_t1 = self.all_data[two_actors[1]][take_combo[0]]["chunks"]
                            ast_keys_a2_t2 = self.all_data[two_actors[1]][take_combo[1]]["chunks"]
                            min_chunks = min(ast_keys_a1_t1, ast_keys_a1_t2, ast_keys_a2_t1, ast_keys_a2_t2)
                            for i in range(min_chunks):
                                
                                assert self.all_data[two_actors[0]][take_combo[0]][f"ast_{i}"]["emo_id"] == \
                                    self.all_data[two_actors[0]][take_combo[1]][f"ast_{i}"]["emo_id"] == \
                                    self.all_data[two_actors[1]][take_combo[0]][f"ast_{i}"]["emo_id"] == \
                                    self.all_data[two_actors[1]][take_combo[1]][f"ast_{i}"]["emo_id"], \
                                    f"Emotion IDs are not equal for {two_actors[0]} and {two_actors[1]}"
                                    
                                assert self.all_data[two_actors[0]][take_combo[0]][f"ast_{i}"]["actor_id"] == \
                                    self.all_data[two_actors[0]][take_combo[1]][f"ast_{i}"]["actor_id"], \
                                    f"Actor IDs are not equal for {two_actors[0]}"
                                    
                                assert self.all_data[two_actors[1]][take_combo[0]][f"ast_{i}"]["actor_id"] == \
                                    self.all_data[two_actors[1]][take_combo[1]][f"ast_{i}"]["actor_id"], \
                                    f"Actor IDs are not equal for {two_actors[1]}"
                                
                                processed_data.append({
                                    "fbank_a1_t1": self.all_data[two_actors[0]][take_combo[0]][f"ast_{i}"]["fbank"], # replaced by a1_t1 in dataload.py
                                    "fbank_a1_t2": self.all_data[two_actors[0]][take_combo[1]][f"ast_{i}"]["fbank"], # replaced by a1_t2
                                    "fbank_a2_t1": self.all_data[two_actors[1]][take_combo[0]][f"ast_{i}"]["fbank"], # replaced by a2_t1
                                    "fbank_a2_t2": self.all_data[two_actors[1]][take_combo[1]][f"ast_{i}"]["fbank"], # replaced by a2_t2
                                    "emo_id": self.all_data[two_actors[0]][take_combo[0]][f"ast_{i}"]["emo_id"],
                                    "a1_id": self.all_data[two_actors[0]][take_combo[0]][f"ast_{i}"]["actor_id"],
                                    "a2_id": self.all_data[two_actors[1]][take_combo[0]][f"ast_{i}"]["actor_id"]                          
                                })
                disentagler_loader[k] = processed_data
            
            np.savez(disentagler_loader_file, train=disentagler_loader["train"], val=disentagler_loader["val"])
            print("[AST] Everything processed!")
        
        else: 
            print("[AST] Disentagler loader already exists, loading...")
            # loader_start_time = time.time()
            disentagler_loader = np.load(disentagler_loader_file, allow_pickle=True)
            # print(f"[AST] Precreated disentagler loader loaded in {(time.time()-loader_start_time)/60} minutes")
        
        print(f"[AST] Training datapoints: {len(disentagler_loader['train'])}, Validation datapoints: {len(disentagler_loader['val'])}") # 12634, 160
        
        if any([self.config["TRAIN_PARAM"]["wav_dtw_mfcc"]["dataset_mean"] == 0.0, self.config["TRAIN_PARAM"]["wav_dtw_mfcc"]["dataset_std"] == 0.0]):
            print("[AST] Computing dataset mean and std...")
            train_fbank = [i["fbank_a1_t1"] for i in disentagler_loader["train"]] + \
                          [i["fbank_a1_t2"] for i in disentagler_loader["train"]] + \
                          [i["fbank_a2_t1"] for i in disentagler_loader["train"]] + \
                          [i["fbank_a2_t2"] for i in disentagler_loader["train"]]
            train_fbank = torch.stack(train_fbank)
            self.config["TRAIN_PARAM"]["wav_dtw_mfcc"]["dataset_mean"] = train_fbank.mean()
            self.config["TRAIN_PARAM"]["wav_dtw_mfcc"]["dataset_std"] = train_fbank.std()
            # raise Exception(f"[AST] Dataset mean: {self.config['TRAIN_PARAM']['wav_dtw_mfcc']['dataset_mean']}, Dataset std: {self.config['TRAIN_PARAM']['wav_dtw_mfcc']['dataset_std']}") # -9.173025, 5.062332
        
        return disentagler_loader

    def latent_diffusion_dm_v2(self, device, verbose=False, audio_ablation=None):
        assert self.smplx_data, "Smplx data flag is False!"
        actors, actors_to_exclude = list(range(1, 31)), [11,20,24,25,27]
        actors = [str(a) for a in actors if a not in actors_to_exclude]
        pretrained_takes = ["9", "10", "65", "66", "73", "74", "81", "82", "87", "88", "95", "96", "103", "104", "111", "112"]

        if self.process_latdiff_cache:
            target_length = self.config["TRAIN_PARAM"]["wav_dtw_mfcc"]["target_length"]
            norm_mean = self.config["TRAIN_PARAM"]["wav_dtw_mfcc"]["dataset_mean"]
            norm_std = self.config["TRAIN_PARAM"]["wav_dtw_mfcc"]["dataset_std"]
            start_time = time.time()
            all_processed_motions = [str(p) for p in Path(self.smplx_extract_path).glob("*.npz")] # 1936 # MOSHed: 624
            processed_motions = [p for p in all_processed_motions if Path(p).stem.split("_")[0] in actors and \
                                                                     Path(p).stem.split("_")[-1] in pretrained_takes] # V0: 257 (Missing: 143)
            processed_motions.sort(key=lambda x: (int(x.split("/")[-1].split("_")[0]), int(x.split("/")[-1].split("_")[-1].split(".")[0])))
            if verbose: print(f"[LPDM-SMPLX] Total number of motion clips: {len(processed_motions)}")
            all_wav = [str(f.resolve()) for f in self.data_path.rglob('*/*.wav')]
            all_wav = [f for f in all_wav if Path(f).stem.split("_")[0] in actors and \
                                             Path(f).stem.split("_")[-1] in pretrained_takes]
            all_wav.sort(key=lambda x: (int(x.split("/")[-1].split("_")[0]), int(x.split("/")[-1].split("_")[-1].split(".")[0])))
            all_wav = [f for f in all_wav if Path(f).stem.split("_")[2] == "0" and \
                                             Path(f).parents[0].stem == Path(f).stem.split("_")[0]]
            if verbose: print(f"[LPDM-SMPLX] Total number of WAV files: {len(all_wav)}")
            if self.smplx_data_type == "ARP": 
                assert len(all_wav) == len(processed_motions) == 252, f"[LPDM-SMPLX] len all_wav: {len(all_wav)}, len processed_motions: {len(processed_motions)}"
                src_fps = 120
            elif self.smplx_data_type == "MOSH":
                
                # FIXME 
                expected_matches = 314 if ("moshed_v1" in str(self.smplx_extract_path)) else 236
                
                processed_motions = [p for p in processed_motions if Path(p).stem in [Path(f).stem for f in all_wav]]
                all_wav = [f for f in all_wav if Path(f).stem in [Path(p).stem for p in processed_motions]]
                assert len(all_wav) == len(processed_motions) == expected_matches, f"[LPDM-SMPLX] len all_wav: {len(all_wav)}, len processed_motions: {len(processed_motions)}, expected_matches: {expected_matches}"
                src_fps = 30
            
            print(f"[LPDM-SMPLX] (1/5) Processing Motion and attributes...")
            frames = 0
            for motion in processed_motions:
                npz_name = str(Path(motion).stem)
                if verbose: print(f"[LPDM-SMPLX] Processing {npz_name}...")
                actor, take = npz_name.split("_")[1], "_".join(npz_name.split("_")[2:])
                load_npz = np.load(motion, allow_pickle=True)
                poses, trans, framerate = load_npz["poses"], load_npz["trans"], load_npz["mocap_frame_rate"]
                trans = np.expand_dims(trans, axis=1)  
                if self.smplx_data_type == "MOSH": poses = poses.reshape(poses.shape[0], -1, 3)
                if poses.shape[0] == trans.shape[0]: motion = np.append(poses, trans, axis=1)
                else:
                    min_frames = min(poses.shape[0], trans.shape[0])
                    motion = np.append(poses[:min_frames], trans[:min_frames], axis=1) 
                self.all_data[actor][take]["ld_motion"] = motion.reshape(motion.shape[0], -1)
                self.all_data[actor][take]["ld_attr"] = self.actor_attr[npz_name.split("_")[0]]
                frames += motion.shape[0]
            if verbose: print(f"[LPDM-SMPLX] Total duration of motion loaded: {frames / 108000} hours") # V0: 4.14 hrs, V1: 5.71 hrs
            
            print(f"[LPDM-SMPLX] (2/5) Processing Emo-labels...")
            for i in tqdm(self.all_data.keys(), desc="[LPDM-SMPLX] Emo-labels for all actors", leave=False):
                for j in self.all_data[i].keys():
                    emo_file = self.all_data[i][j]["emo"][0]
                    emo = torch.from_numpy(np.genfromtxt(emo_file, delimiter=",")[-2:])
                    self.all_data[i][j]["ld_emo_label"] = emo[-1].long()    
            
            print(f"[LPDM-SMPLX] (3/5) Processing Audio...")
            ast = Pretrained_AST_EVP(device)
            ast.get_model(self.config, self.processed, self.tag, audio_ablation=audio_ablation)   
            for wav_file in tqdm(all_wav, desc="[LPDM-SMPLX] Loading audios...", total=len(all_wav), leave=False):
                x_actor_take = wav_file.split("/")[-1].split(".")[0]
                actor, take = x_actor_take.split("_")[1], "_".join(x_actor_take.split("_")[2:])
                if self._emotion_dm_v1_helper(actor) in actors and take.split("_")[-1] in pretrained_takes:
                    self.all_data[actor][take]["ld_audio"] = AudioSegment.from_wav(wav_file)
                    waveform, sr = torchaudio.load(wav_file)
                    assert sr == 16000, f"SR is {sr}, not 16000 for {wav_file}"
                    waveform = waveform - waveform.mean()
                    total_chunks = waveform.shape[1]//160000
                    con, emo, sty = [], [], []
                    for k in range(0, total_chunks):
                        sliced_chunk = waveform[:, k:k+160000]
                        fbank = torchaudio.compliance.kaldi.fbank(sliced_chunk, htk_compat=True, sample_frequency=sr, use_energy=False, window_type='hanning', 
                                                                num_mel_bins=self.config["TRAIN_PARAM"]["wav_dtw_mfcc"]["num_mel_bins"], dither=0.0, frame_shift=10)
                        n_frames = fbank.shape[0]
                        p = target_length - n_frames
                        if p > 0:
                            m = torch.nn.ZeroPad2d((0, 0, 0, p))
                            fbank = m(fbank)
                        elif p < 0: fbank = fbank[0:target_length, :]
                        fbank = (fbank - norm_mean) / (norm_std * 2)
                        ast_feats = ast.get_features(fbank.unsqueeze(0)) # with batch dummy dim
                        con.append(ast_feats["con"].squeeze(0))
                        emo.append(ast_feats["emo"].squeeze(0))
                        sty.append(ast_feats["sty"].squeeze(0))
                        
                    self.all_data[actor][take]["ld_audio_con"] = torch.stack(con, dim=0).cpu().numpy()
                    self.all_data[actor][take]["ld_audio_emo"] = torch.stack(emo, dim=0).cpu().numpy()
                    self.all_data[actor][take]["ld_audio_sty"] = torch.stack(sty, dim=0).cpu().numpy()
            
            print(f"[LPDM-SMPLX] (4/5) Slicing train features...")
            sld_motion, sld_attr, sld_emo_label, sld_audio, sld_audio_con, sld_audio_emo, sld_audio_sty = ([] for _ in range(7))
            sliced_datapoints = 0
            for i in tqdm(self.all_data.keys(), desc="[LPDM-SMPLX] Slicing features for all actors", leave=False):
                for j in tqdm(self.all_data[i].keys(), desc="[LPDM-SMPLX] Slicing features for all takes", leave=False):
                    if self._emotion_dm_v1_helper(i) in actors and j.split("_")[-1] in pretrained_takes and j.split("_")[0] == "0":
                        
                        skip = True if "ld_motion" not in self.all_data[i][j] else False
                        if not skip:
                            motion = self.all_data[i][j]["ld_motion"]
                            attr = self.all_data[i][j]["ld_attr"]
                            emo_label = self.all_data[i][j]["ld_emo_label"]
                            audio = self.all_data[i][j]["ld_audio"]
                            audio_con = self.all_data[i][j]["ld_audio_con"]
                            audio_emo = self.all_data[i][j]["ld_audio_emo"]
                            audio_sty = self.all_data[i][j]["ld_audio_sty"]
                            num_divisions = motion.shape[0] // self.train_pose_framelen
                            assert audio_con.shape[0] == audio_emo.shape[0] == audio_sty.shape[0], \
                                   f"audio_con: {audio_con.shape}, audio_emo: {audio_emo.shape}, audio_sty: {audio_sty.shape}, num_divisions: {num_divisions}"
                            if num_divisions != audio_con.shape[0]:
                                num_divisions = min(num_divisions, audio_con.shape[0])
                                print(f"num_division altered to {num_divisions} for {i}/{j} from {motion.shape[0] // self.train_pose_framelen}, with audio_con: {audio_con.shape[0]}")
                            if verbose: print(f"[LPDM-SMPLX], subdivs: {num_divisions}, motion frames: {motion.shape[0]}, audio_con frames: {audio_con.shape[0]}, audio_emo frames: {audio_emo.shape[0]}, audio_sty frames: {audio_sty.shape[0]}")
                            for pp in range(num_divisions):
                                sld_motion.append(motion[pp*self.train_pose_framelen:(pp+1)*self.train_pose_framelen])
                                sld_attr.append(attr)
                                sld_emo_label.append(emo_label)
                                sld_audio_con.append(audio_con[pp])
                                sld_audio_emo.append(audio_emo[pp])
                                sld_audio_sty.append(audio_sty[pp])
                                current_audio = audio[pp*10000:(pp+1)*10000]
                                current_audio = current_audio.set_frame_rate(16000)
                                channel_sounds = current_audio.split_to_mono()
                                samples = [s.get_array_of_samples() for s in channel_sounds]
                                fp_arr = np.array(samples).T.astype(np.float32)
                                fp_arr /= np.iinfo(samples[0].typecode).max
                                sld_audio.append(fp_arr)
                                sliced_datapoints += 1
                                if verbose:
                                    print(f"[LPDM-SMPLX] (4/5) sld_motion shape: {motion[pp*self.train_pose_framelen:(pp+1)*self.train_pose_framelen].shape}")
                                    print(f"[LPDM-SMPLX] (4/5) sld_audio_con shape: {audio_con[pp].shape}")
                                    print(f"[LPDM-SMPLX] (4/5) sld_audio_emo shape: {audio_emo[pp].shape}")
                                    print(f"[LPDM-SMPLX] (4/5) sld_audio_sty shape: {audio_sty[pp].shape}")
                                    print(f"[LPDM-SMPLX] (4/5) sld_audio shape: {fp_arr.shape}")
            if verbose: print(f"[LPDM-SMPLX] (4/5) Sliced train {sliced_datapoints} datapoints, total time: {sliced_datapoints * 10/3600} hours.") # V0: 3.87 hrs, V1: 5.22 hrs
            
            print(f"[LPDM-SMPLX] (5/5) Creatin train LMDB...")
            map_size = int(1024 * 1024 * 2048 * 4) * 2 * 100
            dst_lmdb_env = lmdb.open(str(self.new_latdiff_cache), map_size=map_size, lock=False) 
            with dst_lmdb_env.begin(write=True) as txn: 
                for s_motion, s_attr, s_emo_label, s_audio, s_audio_con, s_audio_emo, s_audio_sty in zip(sld_motion,
                                                                                                         sld_attr,
                                                                                                         sld_emo_label,
                                                                                                         sld_audio,
                                                                                                         sld_audio_con,
                                                                                                         sld_audio_emo,
                                                                                                         sld_audio_sty):
                            k = "{:005}".format(self.n_latdiff_out_samples).encode("ascii")
                            v = [s_motion, s_attr, s_emo_label, s_audio, s_audio_con, s_audio_emo, s_audio_sty]
                            v = [vv.numpy() if all([isinstance(vv, torch.Tensor)]) else vv for vv in v]
                            v = pa.serialize(v).to_buffer()
                            txn.put(k, v)
                            self.n_latdiff_out_samples += 1 
            with dst_lmdb_env.begin() as txn:
                print(f"[LPDM-SMPLX] LMDB no. of samples: {txn.stat()['entries']}")
            dst_lmdb_env.sync()
            dst_lmdb_env.close()
            
            eval_time = round(time.time()-start_time)/60            
            print("[LPDM-SMPLX-LMDB] Everything processed in {} minutes!".format(eval_time))
        
        else: print("[LPDM-SMPLX-LMDB] Skipping entire latent_diffusion_dm_v2(), lmdbs precreated!")
        
        cache2dl = self.new_latdiff_cache if self.process_latdiff_cache else self.old_latdiff_cache
        if self.backup_cfg is not None: cache2dl = Path(self.backup_cfg["lmdb_cache"])
        return cache2dl   

    def latent_diffusion_eval_dm_v1(self, smplx_data_training=False):
        
        eval_dm = {}
        
        assert self.tag == "latent_diffusion" and self.config["TRAIN_PARAM"]["pretrained_infer"] == True, f"LPDM Infer, Tag: {self.tag}, pretrained_infer: {self.config['TRAIN_PARAM']['pretrained_infer']}"
        style_transfer = self.config["TRAIN_PARAM"]["test"]["style_transfer"]["use"]
        style_cross_emotion_transfer = self.config["TRAIN_PARAM"]["test"]["style_Xemo_transfer"]["use"]
        emotion_control = self.config["TRAIN_PARAM"]["test"]["emotion_control"]["use"]
        content_control = self.config["TRAIN_PARAM"]["test"]["content_control"]["use"]
        
        actors, actors_to_exclude = list(range(1, 31)), [11,20,24,25,27]
        # actors, actors_to_exclude = list(range(1, 31)), [11,20,24,25]    # test
        emotional_actors = [str(a) for a in actors if a not in actors_to_exclude]
        # pretrained_takes = ["65", "66", "73", "74", "81", "82", "87", "88", "95", "96", "103", "104", "111", "112"]
        pretrained_takes = ["9", "10", "65", "66", "73", "74", "81", "82", "87", "88", "95", "96", "103", "104", "111", "112"]
        if not "moshed_v1" in str(self.smplx_extract_path): self.smplx_extract_path = self.config["DATA_PARAM"]["Bvh"]["mosh_extract_path_v1"]
        all_processed_motions = [str(p) for p in Path(self.smplx_extract_path).glob("*.npz")]                                # 1620
        processed_motions = [p for p in all_processed_motions if Path(p).stem.split("_")[0] in emotional_actors and \
                                                                    Path(p).stem.split("_")[-1] in pretrained_takes]            # 330
        motion_npzs = processed_motions
        
        all_wav = [str(f.resolve()) for f in self.data_path.rglob('*/*.wav')]                                                    # 1946
        all_wav = [f for f in all_wav if Path(f).stem in [Path(p).stem for p in processed_motions]]                              # 331
        all_wav = [f for f in all_wav if "21/1_wayne_0_103_103" not in f]   # 21/1_wayne_0_103_103.wav misplaced                 # 330
        
        if style_transfer:
            print(f"[LPDM-EVAL] Processing style_transfer...")
            actors = self.config["TRAIN_PARAM"]["test"]["style_transfer"]["actors"]
            emotion = self.config["TRAIN_PARAM"]["test"]["style_transfer"]["emotion"]
            data_dict = style_transfer_dict(actors, emotion)
        
            info, dd = self._preprare_data_dict_v1(data_dict, all_wav, motion_npzs)
            eval_dm["style_transfer"] = dd
            eval_dm["style_transfer_info"] = info
        
        if style_cross_emotion_transfer:
            print(f"[LPDM-EVAL] Processing style_cross_emotion_transfer...")
            actors = self.config["TRAIN_PARAM"]["test"]["style_Xemo_transfer"]["actors"]
            emotion = self.config["TRAIN_PARAM"]["test"]["style_Xemo_transfer"]["emotion"]
            data_dict = style_Xemo_transfer_dict(actors, emotion)
            
            info, dd = self._preprare_data_dict_v1(data_dict, all_wav, motion_npzs)
            eval_dm["style_Xemo_transfer"] = dd
            eval_dm["style_Xemo_transfer_info"] = info
        
        if emotion_control:
            print(f"[LPDM-EVAL] Processing emotion_control...")
            actor = self.config["TRAIN_PARAM"]["test"]["emotion_control"]["actor"]
            content_emotion = self.config["TRAIN_PARAM"]["test"]["emotion_control"]["content_emotion"]
            take_element = self.config["TRAIN_PARAM"]["test"]["emotion_control"]["take_element"]
            data_dict = emotion_control_dict(actor, content_emotion, take_element)
            
            info, dd = self._preprare_data_dict_v1(data_dict, all_wav, motion_npzs)
            eval_dm["emotion_control"] = dd
            eval_dm["emotion_control_info"] = info
            
        if content_control:
            print(f"[LPDM-EVAL] Processing content_control...")
            actor = self.config["TRAIN_PARAM"]["test"]["content_control"]["actor"]
            raise NotImplementedError("Content control not implemented yet!")
        
        return eval_dm
    
    def _bvhframecountcheck(self, bvh_list, fps, verbose):
        inconsistent_bvh = []
        for bvh in bvh_list:
            with open(str(bvh), "r") as src_data:
                src_data_file = src_data.readlines()
                framerate = float(re.findall("\d+\.\d+",src_data_file[430])[0])
                orig_fps = round(1.0/framerate)
                if orig_fps != fps: 
                    inconsistent_bvh.append(bvh)
                    if verbose: print(f"[DIFF/ inconsistent fps] BVH: {bvh} | FPS: {orig_fps}")
        return inconsistent_bvh
    
    def test_dm(self):
        return self.all_data
    
    def _NestedDictValues(self, d):
        for v in d.values():
            if isinstance(v, dict):
                yield from self._NestedDictValues(v)
            else:
                yield v[0]
                
    def _emotion_dm_v1_helper(self, actor):
        for ind, attr in self.actor_attr.items():
            if actor in attr:
                return ind
            
    def _fetch_motion(self, current_data):
        for i, path in enumerate(self.prior_train_npz_paths):
            if current_data in str(path): 
                take_len = self.prior_train_npz_lengths[i]
                return self.prior_train_motion[i][:take_len]
        for i, path in enumerate(self.prior_val_npz_paths):
            if current_data in str(path): 
                take_len = self.prior_val_npz_lengths[i]
                return self.prior_val_motion[i][:take_len]
        for i, path in enumerate(self.prior_test_npz_paths):
            if current_data in str(path): 
                take_len = self.prior_test_npz_lengths[i]
                return self.prior_test_motion[i][:take_len]
        return Exception("Data not found in any of the prior npz paths!")
        
    def _fetch_attr(self, actor):
        for v in self.actor_attr.values():
            if actor in v: return v
    
    def _preprare_data_dict_v1(self, data_dict, all_wav, motion_npz=None):
        info = data_dict["info"]
        data_dict.pop("info")
        for actor in data_dict.keys():
            for take in data_dict[actor].keys():
                current_data = actor + "_" + take
                if motion_npz is None: 
                    raise Exception("motion_npz is found none, this is not part of current implementation!")
                    data_dict[actor][take]["ld_motion"] = self._fetch_motion(current_data)
                else:
                    motion_file = [f for f in motion_npz if current_data in f][0]
                    load_npz = np.load(motion_file, allow_pickle=True)
                    poses, trans = load_npz["poses"], load_npz["trans"]
                    trans = np.expand_dims(trans, axis=1)  
                    poses = poses.reshape(poses.shape[0], -1, 3)
                    if poses.shape[0] == trans.shape[0]: motion = np.append(poses, trans, axis=1)
                    else:
                        min_frames = min(poses.shape[0], trans.shape[0])
                        motion = np.append(poses[:min_frames], trans[:min_frames], axis=1) 
                    data_dict[actor][take]["ld_motion"] = motion.reshape(motion.shape[0], -1).astype(np.float32)
                wav_ind = all_wav.index([f for f in all_wav if current_data in f][0])
                data_dict[actor][take]["ld_wav"] = AudioSegment.from_wav(all_wav[wav_ind])
                waveform, sr = torchaudio.load(all_wav[wav_ind])
                assert sr == 16000, f"SR is {sr}, not 16000 for {all_wav[wav_ind]}"
                waveform = waveform - waveform.mean()
                data_dict[actor][take]["ld_waveform"] = waveform
                emo_file = self.all_data[actor][take]["emo"][0]
                emo = torch.from_numpy(np.genfromtxt(emo_file, delimiter=",")[-2:])
                data_dict[actor][take]["ld_emo_label"] = emo[-1].long()
                data_dict[actor][take]["ld_attr"] = self._fetch_attr(actor)
        return info, data_dict    