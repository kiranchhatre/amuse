
import torch
from torch import nn
        
        
class FaceNet(nn.Module):
    def __init__(self, cfg, arch_type):
        super().__init__()
        
        self.cfg = cfg
        self.arch = arch_type
        self.inplanes = self.cfg["njoints"][self.arch] 
        self.planes = self.cfg["latent_dim"] 
        
        self.net = nn.Sequential(
            BasicBlock(inplanes=self.inplanes, planes=self.planes//2, ker_size=7, stride=1, first_dilation=3,  downsample=True),
            BasicBlock(inplanes=self.planes//2, planes=self.planes//2, ker_size=3, stride=1, first_dilation=1,  downsample=True),
            BasicBlock(inplanes=self.planes//2, planes=self.planes//2, ker_size=3, stride=1, first_dilation=1),
            BasicBlock(inplanes=self.planes//2, planes=self.planes, ker_size=3, stride=1, first_dilation=1,  downsample=True),
        )
    
    def forward(self, x):
        x = x.permute([0,2,1])
        x = self.net(x)
        x = x.permute([0,2,1])
        return x
    
    
class BasicBlock(nn.Module):
    """ From TIMM/ CaMN

    Args:
        nn (_type_): _description_
    """
    
    def __init__(self, inplanes, planes, ker_size, stride=1, downsample=None, cardinality=1, base_width=64,
                 reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.LeakyReLU, norm_layer=nn.BatchNorm1d,
                 attn_layer=None, aa_layer=None, drop_block=None, drop_path=None) -> None:
        super().__init__()
        
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=ker_size, stride=stride, 
                               padding=first_dilation, dilation=dilation, bias=True)
        self.bn1 = nn.BatchNorm1d(planes)
        self.act1 = nn.LeakyReLU(inplace=True)
        
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=ker_size,
                               padding=ker_size//2, dilation=dilation, bias=True)
        self.bn2 = nn.BatchNorm1d(planes)
        self.act2 = act_layer(inplace=True)
        
        if downsample is not None:
            self.downsample = nn.Sequential(
                nn.Conv1d(inplanes, planes, stride=stride, kernel_size=ker_size, padding=first_dilation, dilation=dilation, bias=True),
                norm_layer(planes),)
        else: self.downsample = None
        
        self.stride = stride
        self.dilation = dilation
        self.drop_block = drop_block
        self.drop_path = drop_path
        
    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn2.weight)
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.act2(out)
        
        return out