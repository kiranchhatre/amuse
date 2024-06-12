
import torch
import random
import lmdb
import pickle5
import itertools
import torchaudio
import numpy as np
import pyarrow as pa
from pathlib import Path
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence

class dataload(data.Dataset):
    
    def __init__(self, data, config, data_type, loader_type, shuffle_type):
        self.data = data # Exception, data is cache path if data_type is "latent_diffusion"
        self.config = config
        self.loader_type = loader_type
        self.shuffle_type = shuffle_type 
        self.data_type = data_type
        create_cache_split = None # None by default, True/False for diffusion cache split
        
        self.all_actors = ['yingqing', 'miranda', 'sophie', 'itoi', 'kieks', 
                        'luqi', 'carla', 'goto', 'li', 'reamey', 'jorge', 
                        'scott', 'katya', 'stewart', 'hailing', 'jaime', 
                        'lu', 'nidal', 'zhao', 'hanieh', 'carlos', 'lawrence', 
                        'ayana', 'daiki', 'solomon', 'kexin', 'tiffnay', 
                        'catherine', 'zhang', 'wayne']

        # 1. Emotion Classification >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        if self.shuffle_type == "random":
            if config["TRAIN_PARAM"][data_type]["balance_neutral"] == True: # exclude extra 5 neutral takes
                self.all_takes = ['0_9_9', '0_10_10', '0_65_65', '0_66_66','0_73_73', '0_74_74',
                              '0_81_81', '0_82_82', '0_87_87', '0_88_88', '0_95_95',
                              '0_96_96', '0_103_103', '0_104_104', '0_111_111', '0_112_112']
            else:
                self.all_takes = ['0_9_9', '0_10_10', '0_11_11', '0_13_13', '0_14_14',
                              '0_15_15', '0_16_16', '0_65_65', '0_66_66','0_73_73', '0_74_74',
                              '0_81_81', '0_82_82', '0_87_87', '0_88_88', '0_95_95',
                              '0_96_96', '0_103_103', '0_104_104', '0_111_111', '0_112_112']
            
            all_act_tak = [(x,y) for x in self.all_actors for y in self.all_takes]
            random.shuffle(all_act_tak)
            self.train_list = all_act_tak[:int(len(all_act_tak)*0.8)]
            self.val_list = all_act_tak[int(len(all_act_tak)*0.8):int(len(all_act_tak)*0.9)]
            self.test_list = all_act_tak[int(len(all_act_tak)*0.9):]
            
        elif self.shuffle_type == "takes":
            # TODO: if config["TRAIN_PARAM"][data_type]["balance_neutral"] == True: # exclude extra 5 neutral takes
            self.train_takes = ['0_9_9', '0_10_10', '0_11_11', '0_13_13', '0_14_14', # neutral
                            '0_65_65', '0_66_66', # happy
                            '0_73_73', '0_74_74', # angry
                            '0_81_81', '0_82_82', # sad
                            '0_87_87', # contempt
                            '0_95_95', # surprise
                            '0_103_103', # fear
                            '0_111_111'] # disgust # 15 takes
            self.val_takes = ['0_15_15', '0_88_88', '0_96_96'] # neutral, contempt, surprise # 3 takes
            self.test_takes = ['0_16_16', '0_104_104', '0_112_112'] # neutral, fear, disgust # 3 takes
            
        elif self.shuffle_type == "actors":
            # TODO: if config["TRAIN_PARAM"][data_type]["balance_neutral"] == True: # exclude extra 5 neutral takes
            self.train_actors = ['yingqing', 'miranda', 'sophie', 'itoi', 'kieks', 
                               'luqi', 'carla', 'goto', 'li', 'reamey', 'jorge', 
                               'scott', 'katya', 'stewart', 'hailing', 'jaime', 
                               'lu', 'nidal', 'zhao', 'hanieh', 'carlos', 'lawrence', 
                               'ayana', 'daiki']
            self.train_actors_v1 = ["wayne", "scott", "solomon", "lawrence", "stewart", 
                                    "carla", "sophie", "catherine", "miranda", "kieks", 
                                    "zhao", "lu", "jorge", "daiki", "ayana", "reamey", 
                                    "yingqing", "katya"]
            self.val_actors = self.config["TRAIN_PARAM"]["val_actors"]
            self.test_actors = self.config["TRAIN_PARAM"]["test_actors"]
            self.val_actors_v1 = self.config["TRAIN_PARAM"]["val_actors_v1"]
            self.test_actors_v1 = self.config["TRAIN_PARAM"]["test_actors_v1"]
            self.emotional_takes = ["0_9_9", "0_10_10", "0_65_65", "0_66_66", "0_73_73", "0_74_74", "0_81_81", "0_82_82", "0_87_87", 
                                    "0_88_88", "0_95_95", "0_96_96", "0_103_103", "0_104_104", "0_111_111", "0_112_112"]         
        
        # 2. DTW Disentanglement >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        if self.data_type not in ["diffusion", "motionprior", "motionprior_long", "latent_diffusion"]:
            if config["TRAIN_PARAM"][data_type]["balance_neutral"] == True:
                self.n_takes = ['0_9_9', '0_10_10']
            else: 
                self.n_takes = ['0_9_9', '0_10_10', '0_11_11', '0_13_13', '0_14_14', '0_15_15', '0_16_16']
            self.h_takes = ['0_65_65', '0_66_66']       # happy 65 - 72
            self.a_takes = ['0_73_73', '0_74_74']       # angry 73 - 80
            self.s_takes = ['0_81_81', '0_82_82']       # sad 81 - 86
            self.c_takes = ['0_87_87', '0_88_88']       # contempt 87 - 94
            self.su_takes = ['0_95_95', '0_96_96']      # surprise 95 - 102
            self.f_takes = ['0_103_103', '0_104_104']   # fear 103 - 110
            self.d_takes = ['0_111_111', '0_112_112']   # disgust 111 - 118
            self.emo_sorted_takes = {
                tuple(self.n_takes): 2189,
                tuple(self.h_takes): 1470,
                tuple(self.a_takes): 1533,
                tuple(self.s_takes): 1689,
                tuple(self.c_takes): 1970,
                tuple(self.su_takes): 1376,
                tuple(self.f_takes): 1720,
                tuple(self.d_takes): 1689            
            }
            self.two_actor_list = list(itertools.combinations(self.all_actors, 2)) # 435 ['solomon', 'kexin'], ['solomon', 'tiffnay'], ...
            self.t_two_actor_list = list(itertools.combinations(self.train_actors, 2))
            self.v_two_actor_list = list(itertools.combinations(self.val_actors, 2))
            self.te_two_actor_list = list(itertools.combinations(self.test_actors, 2))
            self.t_two_actor_list_v1 = list(itertools.combinations(self.train_actors_v1, 2))
            self.v_two_actor_list_v1 = list(itertools.combinations(self.val_actors_v1, 2))
            self.te_two_actor_list_v1 = list(itertools.combinations(self.test_actors_v1, 2))
            
            # AST based
            self.fbank_noise = self.config["TRAIN_PARAM"][self.data_type]["noise"]
            self.fbanks_list = ["fbank_a1_t1", "fbank_a1_t2", "fbank_a2_t1", "fbank_a2_t2"]
            self.norm_mean = self.config["TRAIN_PARAM"][self.data_type]["dataset_mean"]
            self.norm_std = self.config["TRAIN_PARAM"][self.data_type]["dataset_std"]
            freq_mask_param = self.config["TRAIN_PARAM"][self.data_type]["freqm"] if self.loader_type == "train" else 0
            time_mask_param = self.config["TRAIN_PARAM"][self.data_type]["timem"] if self.loader_type == "train" else 0
            self.freqm = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param)
            self.timem = torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param)

        if self.data_type == "latent_diffusion":
            latdiff_cache_path = self.data
            if self.loader_type == "val": 
                # latdiff_cache_path = latdiff_cache_path.parent / latdiff_cache_path.name.replace("_250", "_250_val")
                latdiff_cache_path = latdiff_cache_path.parent / latdiff_cache_path.name.replace("_300", "_300_val") 
            data_file = latdiff_cache_path / "data.mdb"
            assert data_file.is_file(), f"LMDB file not found at {data_file}, please create it first."
            # lmdb env name shared with prior
            self.prior_lmdb_env = lmdb.open(str(latdiff_cache_path), readonly=True, lock=False, readahead=False, meminit=False)
            
        self.setup(create_cache_split)
            
    def setup(self, create_cache_split=None):
        if self.data_type == "wav_dtw_mfcc":
            # return self.audio_disentangler()
            self.data = self.data["train"] if self.loader_type == "train" else self.data["val"]
            self.data = self._fix_person_ID(self.data, self.loader_type)
            return None 
        elif self.data_type in ["motionprior", "motionprior_long", "latent_diffusion"]:
            return None
        elif self.data_type == "test":
            return self.test()
    
    def audio_disentangler(self):
        
        raise Exception("This part is obsolete, not used in the current implementation.")
        
        disentagler_loader = []
        
        if self.config["TRAIN_PARAM"][self.data_type]["use_triplet"]:
            raise NotImplementedError("Not implemented yet")
        
        else:
            
            training_version = self.config["TRAIN_PARAM"][self.data_type]["training_version"]
            if self.shuffle_type == "random":
                raise NotImplementedError("Not implemented yet")
            elif self.shuffle_type == "takes":
                raise NotImplementedError("Not implemented yet")
            elif self.shuffle_type == "actors":
                if training_version == "v0":
                    a_list = self.t_two_actor_list if self.loader_type == "train" else self.v_two_actor_list
                elif training_version == "v1":
                    a_list = self.t_two_actor_list_v1 if self.loader_type == "train" else self.v_two_actor_list_v1
            elif self.shuffle_type == "all":
                a_list = self.two_actor_list
                
            # Cross and self reconstruction for two takes
            for two_actors in a_list: # eg. ('wayne', 'scott')
                for emo_take in self.emo_sorted_takes.keys(): # single emotion only, eg. self.n_takes ('0_9_9', '0_10_10')
                    dtw_len = self.emo_sorted_takes[emo_take] # 2189
                    takes_combo = list(itertools.combinations(emo_take, 2)) # ['0_9_9', '0_10_10'], ['0_9_9', '0_11_11'], ...
                    for take_combo in takes_combo:
                        a_t_key = two_actors[0] + "_" + two_actors[1] + "_" + \
                                  take_combo[0].split("_")[-1] + "_" + take_combo[-1].split("_")[1] # 'solomon_kexin_9_10'
                        
                        # Sanity 1: original dataset has emotion label errors for yingqing and goto
                        if not any(x in ["yingqing", "goto"] for x in two_actors): 
                            assert self.data[two_actors[0]][take_combo[0]]["emo_label"] == \
                                   self.data[two_actors[0]][take_combo[1]]["emo_label"] == \
                                   self.data[two_actors[1]][take_combo[0]]["emo_label"] == \
                                   self.data[two_actors[1]][take_combo[1]]["emo_label"], \
                                   "Emotion labels are not equal"
                        
                        # Sanity 2: length of dtw mfccs are equal
                        assert len(self.data[two_actors[0]][take_combo[0]][self.data_type]) == \
                               len(self.data[two_actors[0]][take_combo[1]][self.data_type]) == \
                               len(self.data[two_actors[1]][take_combo[0]][self.data_type]) == \
                               len(self.data[two_actors[1]][take_combo[1]][self.data_type]) == dtw_len, \
                               "Length of DTW mfccs are not equal"
                        
                        for i in range(dtw_len):
                            mfcc_label_dict = {}
                            mfcc_label_dict["combo"] = a_t_key 
                            mfcc_label_dict["label"] = self.data[two_actors[0]][take_combo[0]]["emo_label"]
                            mfcc_label_dict["a1_t1"] = self.data[two_actors[0]][take_combo[0]][self.data_type][i]
                            mfcc_label_dict["a1_t2"] = self.data[two_actors[0]][take_combo[1]][self.data_type][i]
                            mfcc_label_dict["a1_ID"] = self.actor_IDs[two_actors[0]]
                            mfcc_label_dict["a2_t1"] = self.data[two_actors[1]][take_combo[0]][self.data_type][i]
                            mfcc_label_dict["a2_t2"] = self.data[two_actors[1]][take_combo[1]][self.data_type][i]
                            mfcc_label_dict["a2_ID"] = self.actor_IDs[two_actors[1]]
                            disentagler_loader.append(mfcc_label_dict)

        self.data = disentagler_loader
        # TODO: save train test val lists in a file
        return self.data
    
    
    def test(self):
        return self.data
    
    def __len__(self):
        if self.data_type == "wav_dtw_mfcc":
            return len(self.data) 
        elif self.data_type in ["motionprior", "motionprior_long", "latent_diffusion"]:
            with self.prior_lmdb_env.begin() as txn:
                self.n_samples = txn.stat()["entries"]
            return self.n_samples
        elif self.data_type == "test":
            return len(self.data)
    
    def __getitem__(self, index):
        
        if self.data_type == "wav_dtw_mfcc": 
            datapoint = self.data[index]
            
            for i in self.fbanks_list:
                datapoint[i] = torch.transpose(datapoint[i], 0, 1)
                datapoint[i] = datapoint[i].unsqueeze(0)
            
            if self.freqm != 0:
                for i in self.fbanks_list: datapoint[i] = self.freqm(datapoint[i])
                
            if self.timem != 0:
                for i in self.fbanks_list: datapoint[i] = self.timem(datapoint[i])
                
            for i  in self.fbanks_list: 
                datapoint[i] = torch.squeeze(datapoint[i], 0)
                datapoint[i] = torch.transpose(datapoint[i], 0, 1)
                datapoint[i] = (datapoint[i] - self.norm_mean) / (self.norm_std * 2)
                
            if self.fbank_noise == True:
                for i in self.fbanks_list: 
                    new_key = f"{i}_noisy"
                    datapoint[new_key] = datapoint[i] + torch.rand(datapoint[i].shape[0], datapoint[i].shape[1]) * np.random.rand() / 10
                    datapoint[new_key] = torch.roll(datapoint[new_key], np.random.randint(-10, 10), 0)
            
            return datapoint
        
        elif self.data_type == "latent_diffusion":
            
            with self.prior_lmdb_env.begin(write=False) as txn:
                key = "{:005}".format(index).encode("ascii")
                sample = txn.get(key)
                sample = pa.deserialize(sample)
                assert len(sample) == 7, f"Latent diffusion should have seven samples, got {len(sample)}"
                s_motion, s_attr, s_emo_label, s_audio, s_audio_con, s_audio_emo, s_audio_sty = sample
                # https://discuss.pytorch.org/t/userwarning-the-given-numpy-array-is-not-writeable/78748/12
                motion = torch.from_numpy(np.copy(s_motion)).float()
                audio_con = torch.from_numpy(np.copy(s_audio_con)).float()
                audio_emo = torch.from_numpy(np.copy(s_audio_emo)).float()
                audio_sty = torch.from_numpy(np.copy(s_audio_sty)).float()
                emo_label = torch.from_numpy(np.copy(s_emo_label)).long()
                return {
                    "ld_motion": motion,
                    "ld_audio": s_audio,
                    "ld_audio_con": audio_con,
                    "ld_audio_emo": audio_emo,
                    "ld_audio_sty": audio_sty,
                    "ld_emo_label": emo_label,
                    "ld_attr": s_attr}
        
        elif self.data_type == "test":
            pass
        
        else:
            raise ValueError(f"Invalid data type: {self.data_type}")
        
    def _fix_person_ID(self, all_data, loader_type):
        print(f"Fixing person ID for {loader_type} data...")
        for e in all_data:
            e["a1_id"] = e["a1_id"] - 1
            e["a2_id"] = e["a2_id"] - 1
        return all_data
                   

def latdiff_long_collate_fn_v1(batch): 
    motion = torch.stack([b["ld_motion"] for b in batch])
    audio_length = [b["ld_audio"].shape[0] for b in batch]
    audio_pad = pad_sequence([torch.from_numpy(np.copy(b["ld_audio"])) for b in batch], batch_first=True)   
    assert motion.shape[0] == audio_pad.shape[0], "Motion and audio batch size mismatch"
    assert all(x.shape[0] == max(audio_length) for x in audio_pad), "Padded audio length not equal"
    audio_con = torch.stack([b["ld_audio_con"] for b in batch])
    audio_emo = torch.stack([b["ld_audio_emo"] for b in batch])
    audio_sty = torch.stack([b["ld_audio_sty"] for b in batch])
    emo_label = torch.stack([b["ld_emo_label"] for b in batch])
    attr = [b["ld_attr"] for b in batch]
    new_batch = {
        "ld_motion": motion,
        "ld_audio": audio_pad,
        "ld_audio_length": torch.from_numpy(np.array(audio_length)),
        "ld_audio_con": audio_con,
        "ld_audio_emo": audio_emo,
        "ld_audio_sty": audio_sty,
        "ld_emo_label": emo_label,
        "ld_attr": attr
    }
    return new_batch
