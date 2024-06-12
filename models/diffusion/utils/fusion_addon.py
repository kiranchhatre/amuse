
import torch
from torch import nn

class FusionNet(nn.Module):
    
    def __init__(self, cfg, fusion_in) -> None:
        super().__init__()
        
        self.cfg = cfg
        self.fusion_mode = cfg["fusion"]["type"]
        self.input_dim = fusion_in
        self.latent_dim = cfg["fusion"][self.fusion_mode]["latent_dim"]
        
        if self.fusion_mode == "CaMN":
            self.fusion = nn.Sequential(
                nn.Linear(self.input_dim, self.latent_dim),
                nn.LeakyReLU(True),
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.LeakyReLU(True)
            )    
        elif self.fusion_mode == "MulT":
            raise NotImplementedError(f"Fusion mode {self.fusion_mode} not implemented yet")
        else:
            raise ValueError(f"Invalid fusion mode: {self.fusion_mode}")
        
    def forward(self, concat_feats):
        
        fusion_seq = self.fusion(concat_feats)
        
        return fusion_seq
    
class AddonNet(nn.Module):
    
    def __init__(self, cfg, mode) -> None:
        super().__init__()
        
        self.cfg = cfg
        self.mode = mode
        self.count = cfg["cond_mode"][self.mode]["count"]
        self.latent_dim = cfg["cond_mode"][self.mode]["latent_dim"]
        
        if self.mode == "emotion":
            
            self.addon_emb = nn.Sequential(
                nn.Embedding(self.count, self.latent_dim),
                nn.Linear(self.latent_dim, self.latent_dim)
            )
            self.add_emb_tail = nn.Sequential(
                nn.Conv1d(self.latent_dim, 8, 9, 1, 4),
                nn.BatchNorm1d(8),
                nn.LeakyReLU(0.3, inplace=True),
                nn.Conv1d(8, 16, 9, 1, 4),
                nn.BatchNorm1d(16),
                nn.LeakyReLU(0.3, inplace=True),
                nn.Conv1d(16, 16, 9, 1, 4),
                nn.BatchNorm1d(16),
                nn.LeakyReLU(0.3, inplace=True),
                nn.Conv1d(16, self.latent_dim, 9, 1, 4),
                nn.BatchNorm1d(self.latent_dim),
                nn.LeakyReLU(0.3, inplace=True),
            )
            
        elif self.mode == "speaker":
            self.addon_emb = nn.Sequential(
                nn.Embedding(self.count, self.latent_dim),
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.LeakyReLU(True)
            )
        else:
            raise ValueError(f"Invalid addon mode: {self.mode}")
        
    def forward(self, addon_feats):
        addon_fusion_seq = self.addon_emb(addon_feats)
        if self.mode == "emotion":
            addon_fusion_seq = addon_fusion_seq.permute([0,2,1])
            addon_fusion_seq = self.add_emb_tail(addon_fusion_seq)
            addon_fusion_seq = addon_fusion_seq.permute([0,2,1])
        
        return addon_fusion_seq