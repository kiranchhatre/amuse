
import pickle
import torch
import einops
import torchaudio
from torch import nn
from pathlib import Path

from models.diffusion.face.fac_util import BasicBlock
from models.diffusion.text.txt_util import TemporalConvNet

class AudioNet(nn.Module):
    
    def __init__(self, cfg, nfeats=None, seq_len=None, tcn_channels=None) -> None:
        super().__init__()
        
        self.nonlinear_proj = cfg["cond_mode"]["audio"]["nonlinear"]
        self.tcn_levels = cfg["cond_mode"]["audio"]["tcn_levels"]
        self.hidden_size = cfg["cond_mode"]["audio"]["latent_dim"]
        self.disentangled_audio = True if any([nfeats == i for i in ["content", "emotion"]]) else False
        self.seq_len = seq_len
        
        if self.disentangled_audio:
            self.input_size = cfg["cond_mode"]["audio"]["dis"][nfeats]
        else:
            if tcn_channels is None:
                self.tcn = TemporalConvNet(num_inputs=28*self.seq_len, num_channels=[self.seq_len] * 1, kernel_size=3, dropout=0.25)        
            else:
                num_channels = [self.seq_len + x*(28*self.seq_len-self.seq_len)/tcn_channels for x in range(tcn_channels)][::-1]
                num_channels = list(map(int, num_channels))
                self.tcn = TemporalConvNet(num_inputs=28*self.seq_len, num_channels=num_channels, kernel_size=3, dropout=0.25)
            self.input_size = cfg["cond_mode"]["audio"]["combined_latent"]
            # self.processed = processed
            # self.channels = cfg["cond_mode"]["audio"]["raw_channel"] # Not used for now
            # self.nfeats = cfg["cond_mode"]["audio"]["raw_feat"] # Not used for now
            # # self.input_size = self.channels * self.nfeats # Not used for now
            # self.input_size = cfg["cond_mode"]["audio"]["combined_latent"]
            # with open(str(Path(self.processed, "eng_data_processed/all_eng_extracted_data.pkl")), "rb") as f:
            #     self.all_data = pickle.load(f)
            # self.audio_encoder = WavEncoder(out_dim=self.input_size) 

        if self.nonlinear_proj:
            self.emb_proj = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.input_size, self.hidden_size)
            )
        else:
            self.emb_proj = nn.Linear(self.input_size, self.hidden_size)
 
    def dynamic(self, name: str, module_class, *args, **kwargs):
        if not hasattr(self, name):
            self.add_module(name, module_class(*args, **kwargs))
        return getattr(self, name)
    
    def forward(self, audio_feats, embproj=True):
    # def forward(self, audio_feats, dialogue_id=None, attr=None, in_seq_len=None, mode=None): # obselete
        
        if self.disentangled_audio:
            
            audio_feat_seq = self.emb_proj(audio_feats) 
        
        else:
            audio_feats = einops.rearrange(audio_feats, "b c h w -> b (c w) h")
            audio_feat_tcn = self.tcn(audio_feats)
            if embproj: audio_feat_seq = self.emb_proj(audio_feat_tcn)
            else: audio_feat_seq = audio_feat_tcn
            
            # obselete 
            # speech_cut_batch = []
            # max_wavform_len = 0
            # for i in range(len(dialogue_id[0])):
            #     # fetch audio file
            #     take = "_".join(dialogue_id[0][i].split("_")[2:])
            #     actor = attr[0][i]
            #     wav_file = self.all_data[actor][take]["wav"][0]
            #     SPEECH_WAVEFORM, _ = torchaudio.load(wav_file, normalize=True, channels_first=True)
                
            #     index = dialogue_id[0][i].split("_")[0]
            #     total_divs = dialogue_id[0][i].split("_")[1]
            #     assert int(index) < int(total_divs), "[DIFF - WAV MODEL] index should be less than total_divs"
                
            #     speechfeat_per_div = SPEECH_WAVEFORM.shape[1] // int(total_divs)
            #     speech_cut = SPEECH_WAVEFORM[:, speechfeat_per_div*int(index):speechfeat_per_div*(int(index)+1)]
                
            #     speech_cut_batch.append(speech_cut)
            #     max_wavform_len = max(max_wavform_len, speech_cut.shape[1])
            
            # speech_cut_batch = [torch.nn.functional.pad(speech_cut, (0, max_wavform_len-speech_cut.shape[1])) for speech_cut in speech_cut_batch]
            # speech_cut_batch = torch.cat(speech_cut_batch, dim=0).to(audio_feats.device)
            # audio_feat_seq = self.audio_encoder(speech_cut_batch)
            # audio_feat_channels = audio_feat_seq.shape[1]
        
            # self.tcn = self.dynamic("tcn",
            #                         TemporalConvNet,
            #                         num_inputs=audio_feat_channels, 
            #                         num_channels=[in_seq_len] * self.tcn_levels, 
            #                         kernel_size=3, 
            #                         dropout=0.25)
            
            # for param in self.tcn.parameters():
            #     param.requires_grad = True
            # self.tcn.to(audio_feats.device)
            
            # if mode == "train":
            #     self.tcn.train()
            # elif mode == "val":
            #     with torch.no_grad():
            #         self.tcn.eval()
                    
            # audio_feat_seq = self.tcn(audio_feat_seq)
            # audio_feat_seq = self.emb_proj(audio_feat_seq) 
        
        return audio_feat_seq 
    
class WavEncoder(nn.Module):
    def __init__(self, out_dim): # out_dim = 128
        super().__init__() #128*1*140844 
        self.out_dim = out_dim
        self.feat_extractor = nn.Sequential( #b = (a+3200)/5 a 
                BasicBlock(1, 32, 15, 5, first_dilation=1600, downsample=True),
                BasicBlock(32, 32, 15, 6, first_dilation=0, downsample=True),
                BasicBlock(32, 32, 15, 1, first_dilation=7, ),
                BasicBlock(32, 64, 15, 6, first_dilation=0, downsample=True),
                BasicBlock(64, 64, 15, 1, first_dilation=7),
                BasicBlock(64, self.out_dim, 15, 6,  first_dilation=0,downsample=True),     
            )
        
    def forward(self, wav_data):
        """ Raw audio encoder

        Args:
            out_dim (int): output dimension, default 128
            wav_data (tensor): shape bs x speechwaveform [1 x 56000]

        Returns:
            tensor: embedding of shape bs x seq_len x out_dim [1 x 53 x 128]
        """
        wav_data = wav_data.unsqueeze(1)  # add channel dim
        out = self.feat_extractor(wav_data)
        return out.transpose(1, 2)  # to (batch x seq x dim)
   
