
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

# from models.diffusion.text.txt_models import TxtNet

class TCNTxtNet(nn.Module):
    def __init__(self, cfg, gpt_version, processed, bs) -> None:
        super().__init__()
        
        self.cfg = cfg
        self.bs = bs
        self.tcn_levels = cfg["cond_mode"]["txt"]["tcn_levels"]
        self.tnet = TxtNet(self.cfg, gpt_version, processed, bs)
        
    def dynamic(self, name: str, module_class, *args, **kwargs):
        if not hasattr(self, name):
            self.add_module(name, module_class(*args, **kwargs))
        return getattr(self, name)
    
    def forward(self, dialogue_id, attr, device, in_seq_len, mode):
        
        txt_feat = self.tnet(dialogue_id, attr, device)
        txt_feat_channels = txt_feat.shape[1]
        
        self.tcn = self.dynamic("tcn",
                                TemporalConvNet,
                                num_inputs=txt_feat_channels, 
                                num_channels=[in_seq_len] * self.tcn_levels, 
                                kernel_size=3, 
                                dropout=0.25)
        
        for param in self.tcn.parameters():
            param.requires_grad = True
        self.tcn.to(device)

        # FIXME: Crappy part, model created in forward pass cant be trained
        if mode == "train":
            self.tcn.train()
        elif mode == "val":
            with torch.no_grad():
                self.tcn.eval()
                  
        upsampled_txt_feat = self.tcn(txt_feat)
        return upsampled_txt_feat
        
class TemporalBlock(nn.Module):
    # TODO: check difference with camn temporal block
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv2d(n_inputs, n_outputs, (1, kernel_size),
                                           stride=stride, padding=0, dilation=dilation))
        self.pad = torch.nn.ZeroPad2d((padding, 0, 0, 0))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv2d(n_outputs, n_outputs, (1, kernel_size),
                                           stride=stride, padding=0, dilation=dilation))
        self.net = nn.Sequential(self.pad, self.conv1, self.relu, self.dropout,
                                 self.pad, self.conv2, self.relu, self.dropout)
        self.downsample = nn.Conv1d(
            n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x.unsqueeze(2)).squeeze(2)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCNModel(nn.Module):
    def __init__(self, num_channels, kernel_size=2, dropout=0.2):
        super(TCNModel, self).__init__()
        self.tcn = TemporalConvNet(
            100, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        # return self.tcn(x) # torch.Size([1, 20, 256])
        # return self.tcn(x)[:, :, -1] # torch.Size([1, 20])
        return self.decoder(self.dropout(self.tcn(x)[:, :, -1])) # torch.Size([1, 1])

if __name__ == "__main__":
    
    # model = TCNModel(num_channels=[20] * 2, kernel_size=3, dropout=0.25)
    # bs = 1
    # x = model(torch.randn(bs, 100, 256))
    # print(x.shape)   
    # channel_list = [2800, 2000, 1200, 400, 100]
    channel_list = [100 + x*(28*100-100)/6 for x in range(6)][::-1]
    channel_list = list(map(int, channel_list))
    # model = TemporalConvNet(num_inputs=2800, num_channels=[100] * 1, kernel_size=3, dropout=0.25)
    model = TemporalConvNet(num_inputs=250, num_channels=channel_list, kernel_size=3, dropout=0.25)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(num_params)
    # x = model(torch.randn(128, 2800, 13))
    x = model(torch.randn(64, 250, 256))
    print(x.shape)
    
