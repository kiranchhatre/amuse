
import torch
from torch import nn

from models.diffusion.utils.time_encoding import (
    Timesteps,
    TimestepEmbedding
)
from models.diffusion.utils.cross_attention import (
    SkipTransformerEncoder,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoderLayer
)
from models.diffusion.utils.temos_util import lengths_to_mask
from models.diffusion.utils.position_encoding import build_position_encoding

class Denoiser(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        
        self.latent_dim = 0
        self.text_encoded_dim = 0
        self.condition = 0
        self.abl_plus = 0
        self.ablation_skip_connection = 0
        self.diffusion_only = 0
        self.arch = 0
        self.pe_type = 0
        
        self.nfeats = 0
        self.ff_size = 0
        self.num_layers = 0
        self.num_heads = 0
        self.dropout = 0
        self.normalize_before = 0
        self.activation = 0
        self.flip_sin_to_cos = 0
        self.return_intermediate_dec = 0
        self.position_embedding = 0
        self.freq_shift = 0
        self.guidance_scale = 0
        self.guidance_uncondp = 0
        self.nclasses = 0
        
        if self.diffusion_only:
            self.pose_proj = nn.Linear(self.latent_dim, self.nfeats)
            self.pose_emdb = nn.Linear(self.nfeats, self.latent_dim)
            
        if self.condition in ["txt"]:
            self.time_proj = Timesteps(self.text_encoded_dim, 
                                       self.flip_sin_to_cos,
                                       self.freq_shift)
            self.time_embedding = TimestepEmbedding(self.text_encoded_dim,
                                                    self.latent_dim)
            
            if self.text_encoded_dim != self.latent_dim:
                self.emb_proj = nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(self.text_encoded_dim, self.latent_dim)
                )
             
        elif self.condition in ["audio"]:
            # USE EmbedEmotion()
            raise NotImplementedError
        
        self.query_pos = build_position_encoding(self.latent_dim, self.position_embedding)
        self.mem_pos = build_position_encoding(self.latent_dim, self.position_embedding)
        
        if self.arch == "trans_enc":
            if self.ablation_skip_connection:
                encoder_layer = TransformerEncoderLayer(
                    self.latent_dim,
                    self.num_heads,
                    self.ff_size,
                    self.dropout,
                    self.activation,
                    self.normalize_before
                )
                encoder_norm = nn.LayerNorm(self.latent_dim)
                self.encoder = SkipTransformerEncoder(encoder_layer, self.num_layers, encoder_norm)
            else:
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=self.latent_dim,
                    nhead=self.num_heads,
                    dim_feedforward=self.ff_size,
                    dropout=self.dropout,
                    activation=self.activation
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, self.num_layers)
        elif self.arch == "trans_dec":
            decoder_layer = TransformerDecoderLayer(
                self.latent_dim,
                self.num_heads,
                self.ff_size,
                self.dropout,
                self.activation,
                self.normalize_before
            )
            decoder_norm = nn.LayerNorm(self.latent_dim)
            self.decoder = TransformerDecoder(
                    decoder_layer, 
                    self.num_layers, 
                    decoder_norm, 
                    self.return_intermediate_dec
            )
        else:
            raise ValueError(f"Architecture {self.arch} not supported")
        
    def forward(self, 
                sample,
                timestep,
                encoder_hidden_states,
                lengths=None,
                **kwargs):
        
        sample = sample.permute(1, 0, 2)
        if lengths not in [None, []]:
            mask = lengths_to_mask(lengths, sample.shape[1])
        
        timesteps = timestep.expand(sample.shape[1]).clone()
        time_emb = self.time_proj(timesteps)
        time_emb = time_emb.to(dtype = sample.dtype)
        time_emb = self.time_embedding(time_emb).unsqueeze(0)
        
        if self.condition in ["txt"]:
            encoder_hidden_states = encoder_hidden_states.permute(1, 0, 2)
            text_emb = encoder_hidden_states
            if self.text_encoded_dim != self.latent_dim:
                text_emb_latent = self.emb_proj(text_emb)
            if self.abl_plus:
                emb_latent = time_emb + text_emb_latent
            else:
                emb_latent = torch.cat((time_emb, text_emb_latent), 0)
        elif self.condition in ["audio"]:
            raise NotImplementedError
        else:
            TypeError(f"Condition {self.condition} not supported")
        
        if self.arch == "trans_enc":
            if self.diffusion_only:
                sample = self.pose_emdb(sample)
                xseq = torch.cat((emb_latent, sample), axis=0)
            else:
                xseq = torch.cat((sample, emb_latent), axis=0)
            xseq = self.query_pos(xseq)
            tokens = self.encoder(xseq)
            
            if self.diffusion_only:
                sample = tokens[emb_latent.shape[0]:]
                sample = self.pose_proj(sample)
                sample[~mask.T] = 0
            else:
                sample = tokens[:sample.shape[0]]
                
        elif self.arch == "trans_dec":
            if self.diffusion_only:
                sample = self.pose_emdb(sample)
            sample = self.query_pos(sample) 
            emb_latent = self.mem_pos(emb_latent)
            sample = self.decoder(tgt=sample, memory=emb_latent).unsqueeze(0)
            
            if self.diffusion_only:
                sample = self.pose_proj(sample)
                sample[~mask.T] = 0
        
        else:
            raise TypeError(f"Architecture {self.arch} not supported")

        sample = sample.permute(1, 0, 2)
        return (sample, )
        
class EmbedEmotion(nn.Module):
    """To embed categorical emotion labels into a continuous space.

    Args:
        nn (_type_): _description_
    """
    def __init__(self):
        pass
    
    def forward(self, x):
        pass

