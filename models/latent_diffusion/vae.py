
import json
import torch
import torch.nn as nn
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Union
from torch.distributions.distribution import Distribution

from models.latent_diffusion.utils import PositionalEncoding
from models.latent_diffusion.utils.temos_utils import lengths_to_mask
from models.latent_diffusion.utils.position_encoding import build_position_encoding
from models.latent_diffusion.utils.cross_attention import (
    SkipTransformerEncoder,
    SkipTransformerDecoder,
    TransformerEncoderLayer,
    TransformerDecoderLayer
)

class MotionPrior(nn.Module):
    
    def __init__(self) -> None:
        super().__init__() 

    def setup(self, processed, config, prior_cfg=None, load_pretrained=False):
        
        self.processed = processed
        self.config = config
        self.smplx_data = self.config["TRAIN_PARAM"]["latent_diffusion"]["smplx_data"]
        self.skip_trans = self.config["TRAIN_PARAM"]["latent_diffusion"]["skip_trans"]
        self.train_upper_body = self.config["TRAIN_PARAM"]["latent_diffusion"]["train_upper_body"]
        self.smplx_rep = self.config["TRAIN_PARAM"]["latent_diffusion"]["smplx_rep"]
        if prior_cfg is not None: self.prior_cfg = prior_cfg
        else:
            if not load_pretrained:
                if self.config["TRAIN_PARAM"]["motionprior"]["emotional"]: 
                    if "_fing" in self.config["TRAIN_PARAM"]["diffusion"]["lmdb_cache"]: cfg_name = "prior_emotional_fing"
                    else: cfg_name = "prior_emotional"
                else: cfg_name = "prior"
                with open(str(Path(self.processed.parents[1], f"configs/{cfg_name}.json")), "r") as f:
                    self.prior_cfg = json.load(f)
            else:
                # Assertions for pretrained model
                assert self.config["TRAIN_PARAM"]["tag"] == "latent_diffusion", "Pretrained model is only available for latent diffusion"
                strs2assert = ["v2", "emotional", "fing", "250"]
                assert all([s in self.config["TRAIN_PARAM"]["diffusion"]["lmdb_cache"] for s in strs2assert]), "Pretrained model is only available for _v2_emotional_fing_250"
                cfg_name = "prior_emotional_fing"
                with open(str(Path(self.processed.parents[1], f"configs/{cfg_name}.json")), "r") as f: self.prior_cfg = json.load(f)
              
        ablation = self.prior_cfg["ablation"]
        nfeats: int = self.prior_cfg["arch_main"]["nfeats"]
        if "_v1_" in self.config["TRAIN_PARAM"]["diffusion"]["lmdb_cache"] and not "ablation" in self.config["TRAIN_PARAM"]["wav_dtw_mfcc"]:
            # ALERT HACK: modified for AMUSEPP (CVPR): 
            # old: only check v1 in cache
            # new: check v1 in cache and confirm that "ablation" doesnt exist in wav_dtw_mfcc (since v1 exists in large BEAT data used for CVPR)
            nfeats -= 3 # remove first order derivatives without roottransforms
        elif self.smplx_data: 
            if self.skip_trans:
                assert self.smplx_rep == "3D", f"[Prior Setup] skip_trans is only available for 3D SMPL-X representation, but got {self.smplx_rep}"
                nfeats -= 36
                print("[Prior Setup] SMPL-X data without root trans, removing 36 features from input")
            elif self.train_upper_body:
                assert self.smplx_rep == "3D", f"[Prior Setup] train_upper_body is only available for 3D SMPL-X representation, but got {self.smplx_rep}"
                nfeats -= 60
                print("[Prior Setup] Training SMPLX upper body only")
            else:
                if self.smplx_rep == "3D": 
                    nfeats -= 33
                    print("[Prior Setup] SMPL-X 3D data detected, removing 33 features from input")
                else:
                    assert self.smplx_rep == "6D", f"[Prior Setup] smplx_rep should be either 3D or 6D, but got {self.smplx_rep}"
                    nfeats += 132
                    print("[Prior Setup] SMPL-X 6D data detected, adding 132 features from input")
                
        latent_dim: list = self.prior_cfg["arch_main"]["latent_dim"]
        ff_size: int = self.prior_cfg["arch_main"]["ff_size"]
        num_layers: int = self.prior_cfg["arch_main"]["num_layers"]
        num_heads: int = self.prior_cfg["arch_main"]["num_heads"]
        dropout: float = self.prior_cfg["arch_main"]["dropout"]
        arch: str = self.prior_cfg["arch_main"]["arch"]
        normalize_before: bool = self.prior_cfg["arch_main"]["normalize_before"]
        activation: str = self.prior_cfg["arch_main"]["activation"]
        position_embedding: str = self.prior_cfg["arch_main"]["position_embedding"]

        self.latent_size = latent_dim[0]
        self.latent_dim = latent_dim[-1]
        input_feats = nfeats
        output_feats = nfeats
        self.arch = arch
        self.mlp_dist = ablation["MLP_DIST"]
        self.pe_type = ablation["PE_TYPE"]

        if self.pe_type == "actor":
            self.query_pos_encoder = PositionalEncoding(
                self.latent_dim, dropout)
            self.query_pos_decoder = PositionalEncoding(
                self.latent_dim, dropout)
        elif self.pe_type == "mld":                                             # FIXME: PositionEmbeddingSine1D
            self.query_pos_encoder = build_position_encoding(
                self.latent_dim, position_embedding=position_embedding)
            self.query_pos_decoder = build_position_encoding(
                self.latent_dim, position_embedding=position_embedding)
        else:
            raise ValueError("Not Support PE type")

        encoder_layer = TransformerEncoderLayer(
            self.latent_dim,
            num_heads,
            ff_size,
            dropout,
            activation,
            normalize_before,
        )
        encoder_norm = nn.LayerNorm(self.latent_dim)
        self.encoder = SkipTransformerEncoder(encoder_layer, num_layers,
                                              encoder_norm)

        if self.arch == "all_encoder":
            decoder_norm = nn.LayerNorm(self.latent_dim)
            self.decoder = SkipTransformerEncoder(encoder_layer, num_layers,
                                                  decoder_norm)
        elif self.arch == "encoder_decoder":
            decoder_layer = TransformerDecoderLayer(
                self.latent_dim,
                num_heads,
                ff_size,
                dropout,
                activation,
                normalize_before,
            )
            decoder_norm = nn.LayerNorm(self.latent_dim)
            self.decoder = SkipTransformerDecoder(decoder_layer, num_layers,
                                                  decoder_norm)
        else:
            raise ValueError("Not support architecture!")

        if self.mlp_dist:
            self.global_motion_token = nn.Parameter(
                torch.randn(self.latent_size, self.latent_dim))
            self.dist_layer = nn.Linear(self.latent_dim, 2 * self.latent_dim)
        else:
            self.global_motion_token = nn.Parameter(
                torch.randn(self.latent_size * 2, self.latent_dim))

        self.skel_embedding = nn.Linear(input_feats, self.latent_dim)
        self.final_layer = nn.Linear(self.latent_dim, output_feats)

    def forward(self, features: Tensor, lengths: Optional[List[int]] = None):
        # z, dist = self.encode(features, lengths)
        # feats_rst = self.decode(z, lengths)
        # return feats_rst, z, dist
        raise Exception("Should Not enter here")

    def encode(
            self,
            features: Tensor,
            lengths: Optional[List[int]] = None
    ) -> Union[Tensor, Distribution]:
        if lengths is None:
            lengths = [len(feature) for feature in features]

        device = features.device

        bs, nframes, nfeats = features.shape
        mask = lengths_to_mask(lengths, device)

        x = features
        # Embed each human poses into latent vectors
        x = self.skel_embedding(x)

        # Switch sequence and batch_size because the input of
        # Pytorch Transformer is [Sequence, Batch size, ...]
        x = x.permute(1, 0, 2)  # now it is [nframes, bs, latent_dim]

        # Each batch has its own set of tokens
        dist = torch.tile(self.global_motion_token[:, None, :], (1, bs, 1))

        # create a bigger mask, to allow attend to emb
        dist_masks = torch.ones((bs, dist.shape[0]),
                                dtype=bool,
                                device=x.device)
        aug_mask = torch.cat((dist_masks, mask), 1)

        # adding the embedding token for all sequences
        xseq = torch.cat((dist, x), 0)

        if self.pe_type == "actor":
            xseq = self.query_pos_encoder(xseq)
            dist = self.encoder(xseq,
                                src_key_padding_mask=~aug_mask)[:dist.shape[0]]
        elif self.pe_type == "mld":
            xseq = self.query_pos_encoder(xseq)
            dist = self.encoder(xseq,
                                src_key_padding_mask=~aug_mask)[:dist.shape[0]]
            # query_pos = self.query_pos_encoder(xseq)
            # dist = self.encoder(xseq, pos=query_pos, src_key_padding_mask=~aug_mask)[
            #     : dist.shape[0]
            # ]

        # content distribution
        # self.latent_dim => 2*self.latent_dim
        if self.mlp_dist:
            tokens_dist = self.dist_layer(dist)
            mu = tokens_dist[:, :, :self.latent_dim]
            logvar = tokens_dist[:, :, self.latent_dim:]
        else:
            mu = dist[0:self.latent_size, ...]
            logvar = dist[self.latent_size:, ...]

        # resampling
        std = logvar.exp().pow(0.5)
        dist = torch.distributions.Normal(mu, std)
        latent = dist.rsample()
        return latent, dist

    def decode(self, z: Tensor, lengths: List[int]):
        mask = lengths_to_mask(lengths, z.device)
        bs, nframes = mask.shape

        queries = torch.zeros(nframes, bs, self.latent_dim, device=z.device)

        # todo
        # investigate the motion middle error!!!

        # Pass through the transformer decoder
        # with the latent vector for memory
        if self.arch == "all_encoder":
            xseq = torch.cat((z, queries), axis=0)
            z_mask = torch.ones((bs, self.latent_size),
                                dtype=bool,
                                device=z.device)
            augmask = torch.cat((z_mask, mask), axis=1)

            if self.pe_type == "actor":
                xseq = self.query_pos_decoder(xseq)
                output = self.decoder(
                    xseq, src_key_padding_mask=~augmask)[z.shape[0]:]
            elif self.pe_type == "mld":
                xseq = self.query_pos_decoder(xseq)
                output = self.decoder(
                    xseq, src_key_padding_mask=~augmask)[z.shape[0]:]
                # query_pos = self.query_pos_decoder(xseq)
                # output = self.decoder(
                #     xseq, pos=query_pos, src_key_padding_mask=~augmask
                # )[z.shape[0] :]

        elif self.arch == "encoder_decoder":
            if self.pe_type == "actor":
                queries = self.query_pos_decoder(queries)
                output = self.decoder(tgt=queries,
                                      memory=z,
                                      tgt_key_padding_mask=~mask).squeeze(0)
            elif self.pe_type == "mld":
                queries = self.query_pos_decoder(queries)
                # mem_pos = self.mem_pos_decoder(z)
                output = self.decoder(
                    tgt=queries,
                    memory=z,
                    tgt_key_padding_mask=~mask,
                    # query_pos=query_pos,
                    # pos=mem_pos,
                ).squeeze(0)
                # query_pos = self.query_pos_decoder(queries)
                # # mem_pos = self.mem_pos_decoder(z)
                # output = self.decoder(
                #     tgt=queries,
                #     memory=z,
                #     tgt_key_padding_mask=~mask,
                #     query_pos=query_pos,
                #     # pos=mem_pos,
                # ).squeeze(0)

        output = self.final_layer(output)
        # zero for padded area
        output[~mask.T] = 0
        # Pytorch Transformer: [Sequence, Batch size, ...]
        feats = output.permute(1, 0, 2)
        return feats
