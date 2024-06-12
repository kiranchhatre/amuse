import torch
import torch.nn as nn
from torch import  nn

from models.latent_diffusion.utils.embeddings import (TimestepEmbedding,
                                                      Timesteps)
from models.latent_diffusion.utils import PositionalEncoding
from models.latent_diffusion.utils.temos_utils import lengths_to_mask
from models.latent_diffusion.utils.position_encoding import build_position_encoding
from models.latent_diffusion.utils.cross_attention import (SkipTransformerEncoder,
                                                           TransformerDecoder,
                                                           TransformerDecoderLayer,
                                                           TransformerEncoder,
                                                           TransformerEncoderLayer)

class Denoiser(nn.Module):

    def __init__(self,
                 denoiser_cfg) -> None:

        super().__init__()

        self.nfeats = denoiser_cfg["nfeats"] # only for v2 pymo processing
        self.smplx_rep = denoiser_cfg["smplx_rep"]
        if "smplx_data" in denoiser_cfg.keys():
            if "skip_trans" in denoiser_cfg.keys():
                assert self.smplx_rep == "3D", f"[Denoiser Setup] skip_trans setup only for 3D, not {self.smplx_rep}"
                print("[Denoiser Setup] SMPL-X data without root trans, removing 36 features from input")
                self.nfeats -= 36
            elif "train_upper_body" in denoiser_cfg.keys():
                assert self.smplx_rep == "3D", f"[Denoiser Setup] train_upper_body setup only for 3D, not {self.smplx_rep}"
                print("[Denoiser Setup] Training SMPLX upper body only")
                self.nfeats -= 60
            else:
                if self.smplx_rep == "3D":
                    print("[Denoiser Setup] SMPL-X 3D data detected, removing 33 features from input")
                    self.nfeats -= 33
                else: 
                    assert self.smplx_rep == "6D", f"[Denoiser Setup] smplx_rep must be 3D or 6D, not {self.smplx_rep}"
                    print("[Denoiser Setup] SMPL-X 6D data detected, adding 132 features from input")
                    self.nfeats += 132
        
        self.latent_dim = denoiser_cfg["latent_dim"][-1]
        self.ff_size = denoiser_cfg["ff_size"]
        self.num_layers = denoiser_cfg["num_layers"]
        self.num_heads = denoiser_cfg["num_heads"]
        self.dropout = denoiser_cfg["dropout"]
        self.guidance_scale = denoiser_cfg["guidance_scale"]
        self.guidance_uncondp = denoiser_cfg["guidance_uncondp"]
        self.arch = denoiser_cfg["arch"]
        self.normalize_before = denoiser_cfg["normalize_before"]
        self.activation = denoiser_cfg["activation"]
        self.position_embedding = denoiser_cfg["position_embedding"]
        self.cond_dim = denoiser_cfg["cond_dim"]
        self.nclasses = denoiser_cfg["nclasses"] # emotions
        self.freq_shift = denoiser_cfg["freq_shift"]
        self.ablation_skip_connection = denoiser_cfg["ablation_skip_connection"]
        self.pe_type = denoiser_cfg["pe_type"]
        self.flip_sin_to_cos = denoiser_cfg["flip_sin_to_cos"]
        self.return_intermediate_dec = denoiser_cfg["return_intermediate_dec"]
        self.diffusion_only = denoiser_cfg["diffusion_only"]
        self.abl_plus = False
        
        if self.diffusion_only:
            self.pose_embd = nn.Linear(self.nfeats, self.latent_dim)
            self.pose_proj = nn.Linear(self.latent_dim, self.nfeats)

        # emb proj
        self.time_proj = Timesteps(self.cond_dim, self.flip_sin_to_cos,
                                    self.freq_shift)
        self.time_embedding = TimestepEmbedding(self.cond_dim,
                                                self.latent_dim)
        
        self.emb_proj_con = nn.Sequential(
            nn.ReLU(), nn.Linear(self.cond_dim, self.latent_dim))
        self.emb_proj_emo = nn.Sequential(
            nn.ReLU(), nn.Linear(self.cond_dim, self.latent_dim))
        self.emb_proj_sty = nn.Sequential(
            nn.ReLU(), nn.Linear(self.cond_dim, self.latent_dim))

        if self.pe_type == "actor":
            self.query_pos = PositionalEncoding(self.latent_dim, self.dropout)
            self.mem_pos = PositionalEncoding(self.latent_dim, self.dropout)
        elif self.pe_type == "mld":
            self.query_pos = build_position_encoding(
                self.latent_dim, position_embedding=self.position_embedding)
            self.mem_pos = build_position_encoding(
                self.latent_dim, position_embedding=self.position_embedding)
        else:
            raise ValueError("Not Support PE type")

        if self.arch == "trans_enc":
            if self.ablation_skip_connection:
                # use DETR transformer
                encoder_layer = TransformerEncoderLayer(
                    self.latent_dim,
                    self.num_heads,
                    self.ff_size,
                    self.dropout,
                    self.activation,
                    self.normalize_before,
                )
                encoder_norm = nn.LayerNorm(self.latent_dim)
                self.encoder = SkipTransformerEncoder(encoder_layer,
                                                      self.num_layers, encoder_norm)
            else:
                # use torch transformer
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=self.latent_dim,
                    nhead=self.num_heads,
                    dim_feedforward=self.ff_size,
                    dropout=self.dropout,
                    activation=self.activation)
                self.encoder = nn.TransformerEncoder(encoder_layer,
                                                     num_layers=self.num_layers)
        elif self.arch == "trans_dec":
            decoder_layer = TransformerDecoderLayer(
                self.latent_dim,
                self.num_heads,
                self.ff_size,
                self.dropout,
                self.activation,
                self.normalize_before,
            )
            decoder_norm = nn.LayerNorm(self.latent_dim)
            self.decoder = TransformerDecoder(
                decoder_layer,
                self.num_layers,
                decoder_norm,
                return_intermediate=self.return_intermediate_dec,
            )
        else:
            raise ValueError(f"Not supported architechure{self.arch}!")

    def forward(self,
                sample,
                timestep,
                con_hidden,
                emo_hidden,
                sty_hidden,
                lengths=None,
                **kwargs):
        
        sample = sample.permute(1, 0, 2)
        mask = lengths_to_mask(lengths, sample.device)
        timesteps = timestep.expand(sample.shape[1]).clone()
        time_emb = self.time_proj(timesteps)
        time_emb = time_emb.to(dtype=sample.dtype)
        time_emb = self.time_embedding(time_emb).unsqueeze(0)
        
        latents = []
        
        con_hidden = con_hidden.permute(1, 0, 2)
        if self.cond_dim != self.latent_dim:
            con_emb_latent = self.emb_proj_con(con_hidden) 
        else: con_emb_latent = con_hidden
        latents.append(con_emb_latent)
        
        if emo_hidden is not None:
            emo_hidden = emo_hidden.permute(1, 0, 2)
            if self.cond_dim != self.latent_dim:
                emo_emb_latent = self.emb_proj_emo(emo_hidden) 
            else: emo_emb_latent = emo_hidden
            latents.append(emo_emb_latent)
        
        if sty_hidden is not None:
            sty_hidden = sty_hidden.permute(1, 0, 2)
            if self.cond_dim != self.latent_dim:
                sty_emb_latent = self.emb_proj_sty(sty_hidden)
            else: sty_emb_latent = sty_hidden
            latents.append(sty_emb_latent)
            
        if self.abl_plus: emb_latent = time_emb + sum(latents)
        else: emb_latent = torch.cat((time_emb, *latents), axis=0)
        
        if self.arch == "trans_enc":
            if self.diffusion_only:
                sample = self.pose_embd(sample)
                xseq = torch.cat((emb_latent, sample), axis=0)
            else: xseq = torch.cat((sample, emb_latent), axis=0)
            xseq = self.query_pos(xseq)
            tokens = self.encoder(xseq)
            
            if self.diffusion_only:
                sample = tokens[emb_latent.shape[0]:]
                sample = self.pose_proj(sample)
                sample[~mask.T] = 0
            else: sample = tokens[:sample.shape[0]]

        elif self.arch == "trans_dec":
            if self.diffusion_only:
                sample = self.pose_embd(sample)
            sample = self.query_pos(sample)
            emb_latent = self.mem_pos(emb_latent)
            sample = self.decoder(tgt=sample, memory=emb_latent).squeeze(0)

            if self.diffusion_only:
                sample = self.pose_proj(sample)
                sample[~mask.T] = 0
                
        else: raise TypeError("{self.arch} is not supoorted")

        sample = sample.permute(1, 0, 2)
        return (sample, )
