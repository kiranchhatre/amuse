import time
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from torch import nn
from torch.autograd import Variable

class EmotionNet(nn.Module):
    def __init__(self):
        super(EmotionNet, self).__init__()
        
        self.nc = 1
        self.nfe = 64
        self.nfd = 64
        self.latent_dim = 256

        self.emotion_eocder = nn.Sequential(
            conv2d(1,64,3,1,1),
            nn.MaxPool2d((1,3), stride=(1,2)), #[1, 64, 12, 12]
            conv2d(64,128,3,1,1),
            conv2d(128,256,3,1,1),
            nn.MaxPool2d((12,1), stride=(12,1)), #[1, 256, 1, 12]
            conv2d(256,512,3,1,1),
            nn.MaxPool2d((1,2), stride=(1,2))) #[1, 512, 1, 6]   
        self.emotion_eocder_fc = nn.Sequential(
            nn.Linear(512 *6,2048),
            nn.ReLU(True),
            nn.Linear(2048,256),
            nn.ReLU(True),)  
        self.encoder = nn.Sequential(
            nn.Conv2d(self.nc, self.nfe, 4, 2, (4,1), bias=False),
            nn.BatchNorm2d(self.nfe),
            nn.LeakyReLU(True),
            nn.Conv2d(self.nfe, self.nfe * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nfe * 2),
            nn.LeakyReLU(True),
            nn.Conv2d(self.nfe * 2, self.nfe * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nfe * 4),
            nn.LeakyReLU(True),
            nn.Conv2d(self.nfe * 4, self.nfe * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nfe * 8),
            nn.LeakyReLU(True),
            nn.Conv2d(self.nfe * 8, self.nfe * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nfe * 16),
            nn.LeakyReLU(True),
            nn.Conv2d(self.nfe * 16, self.latent_dim, 8, 1, 0, bias=False),
            nn.BatchNorm2d(self.latent_dim),
            nn.LeakyReLU(True)
        )
        self.last_fc = nn.Linear(256,8)
    
    def forward(self, mfcc): # torch.Size([16, 1, 12, 28])
    # def forward(self, mfcc, label): # torch.Size([16, 1, 12, 28])
        # batched input: (torch.Size([128, 1, 12, 28]), torch.Size([128]))
        feature = self.emotion_eocder(mfcc) # [16, 512, 1, 6]
        feature = feature.view(feature.size(0),-1) # [16, 3072]
        last_layer = self.emotion_eocder_fc(feature) # [16, 128]
        emofeats = self.encoder(last_layer)
        emofeats = emofeats.squeeze()
        fake = self.last_fc(emofeats) 
        return fake, emofeats

class StyleNet(nn.Module):
    def __init__(self):
        super(StyleNet, self).__init__()
        
        self.nc = 1
        self.nfe = 64
        self.nfd = 64
        self.latent_dim = 256
        
        self.emotion_eocder = nn.Sequential(
            conv2d(1,64,3,1,1),
            nn.MaxPool2d((1,3), stride=(1,2)), #[1, 64, 12, 12]
            conv2d(64,128,3,1,1),
            conv2d(128,256,3,1,1),
            nn.MaxPool2d((12,1), stride=(12,1)), #[1, 256, 1, 12]
            conv2d(256,512,3,1,1),
            nn.MaxPool2d((1,2), stride=(1,2))) #[1, 512, 1, 6]   
        self.emotion_eocder_fc = nn.Sequential(
            nn.Linear(512 *6,2048),
            nn.ReLU(True),
            nn.Linear(2048,256),
            nn.ReLU(True),)  
        self.encoder = nn.Sequential(
            nn.Conv2d(self.nc, self.nfe, 4, 2, (4,1), bias=False),
            nn.BatchNorm2d(self.nfe),
            nn.LeakyReLU(True),
            nn.Conv2d(self.nfe, self.nfe * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nfe * 2),
            nn.LeakyReLU(True),
            nn.Conv2d(self.nfe * 2, self.nfe * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nfe * 4),
            nn.LeakyReLU(True),
            nn.Conv2d(self.nfe * 4, self.nfe * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nfe * 8),
            nn.LeakyReLU(True),
            nn.Conv2d(self.nfe * 8, self.nfe * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nfe * 16),
            nn.LeakyReLU(True),
            nn.Conv2d(self.nfe * 16, self.latent_dim, 8, 1, 0, bias=False),
            nn.BatchNorm2d(self.latent_dim),
            nn.LeakyReLU(True)
        )
        self.last_fc = nn.Linear(256,30)
    
    def forward(self, mfcc): # torch.Size([16, 1, 12, 28])
    # def forward(self, mfcc, label): # torch.Size([16, 1, 12, 28])
        # batched input: (torch.Size([128, 1, 12, 28]), torch.Size([128]))
        feature = self.emotion_eocder(mfcc) # [16, 512, 1, 6]
        feature = feature.view(feature.size(0),-1) # [16, 3072]
        last_layer = self.emotion_eocder_fc(feature) # [16, 128]
        styfeats = self.encoder(last_layer)
        styfeats = styfeats.squeeze()
        fake = self.last_fc(styfeats)
        return fake, styfeats

class Ct_encoder(nn.Module):
    def __init__(self) -> None:
        super(Ct_encoder, self).__init__()
        
        self.nc = 1
        self.nfe = 64
        self.nfd = 64
        self.latent_dim = 256
        
        self.encoder = nn.Sequential(
            conv2d(1,64,3,1,1),
            conv2d(64,128,3,1,1),
            nn.MaxPool2d(3, stride=(1,2)), 
            conv2d(128,256,3,1,1),
            conv2d(256,256,3,1,1),
            conv2d(256,512,3,1,1),
            nn.MaxPool2d(3, stride=(2,2)))
        self.encoder_fc = nn.Sequential(
            nn.Linear(1024 * 12, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 256),
            nn.ReLU(True))
        self.encoderTCN = nn.Sequential(
            nn.Conv2d(self.nc, self.nfe, 4, 2, (4,1), bias=False),
            nn.BatchNorm2d(self.nfe),
            nn.LeakyReLU(True),
            nn.Conv2d(self.nfe, self.nfe * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nfe * 2),
            nn.LeakyReLU(True),
            nn.Conv2d(self.nfe * 2, self.nfe * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nfe * 4),
            nn.LeakyReLU(True),
            nn.Conv2d(self.nfe * 4, self.nfe * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nfe * 8),
            nn.LeakyReLU(True),
            nn.Conv2d(self.nfe * 8, self.nfe * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nfe * 16),
            nn.LeakyReLU(True),
            nn.Conv2d(self.nfe * 16, self.latent_dim, 8, 1, 0, bias=False),
            nn.BatchNorm2d(self.latent_dim),
            nn.LeakyReLU(True)
        )
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.encoder_fc(x)
        x = x.squeeze()
        return self.encoderTCN(x)

class Decoder(nn.Module):
    def __init__(self) -> None:
        super(Decoder, self).__init__()
        
        self.nc = 1
        self.nfe = 64
        self.nfd = 64
        self.latent_dim = 256
        
        self.decoderTCN = nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim, self.nfd * 16, 8, 1, 0, bias=False),
            nn.BatchNorm2d(self.nfd * 16),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.nfd * 16, self.nfd * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nfd * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.nfd * 8, self.nfd * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nfd * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.nfd * 4, self.nfd * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nfd * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.nfd * 2, self.nfd, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nfd),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.nfd, self.nc, 4, 2, (4,1), bias=False),
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(384, 256, kernel_size=6, stride=2, padding=1, bias=True),
            nn.ConvTranspose2d(256*3, 256, kernel_size=6, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=(4,2), stride=2, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=(4,3), stride=(2,1), padding=(3,1), bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Tanh())
    def forward(self, c, e, s):
        c = self.decoderTCN(c)
        e = self.decoderTCN(e)
        s = self.decoderTCN(s)    
        feat = torch.cat((c,e,s), dim=1)
        feat = torch.unsqueeze(feat, dim=2)
        feat = torch.unsqueeze(feat, dim=3)
        x = 90 * self.decoder(feat)
        return x

class AutoEncoder3x(nn.Module):

    def __init__(self):
        super(AutoEncoder3x, self).__init__()
        
        # models
        self.con_encoder = Ct_encoder()
        self.emo_encoder = EmotionNet()
        self.sty_encoder = StyleNet()
        self.decoder = Decoder()
        
        # losses
        self.CroEn_loss = nn.CrossEntropyLoss()
        self.l1_loss = nn.L1Loss()
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0)
    
    def reconstruct(self, a1_t1, a1_t2, a2_t1, a2_t2):
        
        recons_dict = dict()
        
        recons_dict["c1_1"] = self.con_encoder(a1_t1) # content 1 in actor 1
        recons_dict["c2_1"] = self.con_encoder(a2_t1) # content 1 in actor 2
        recons_dict["c1_2"] = self.con_encoder(a1_t2) # content 2 in actor 1
        recons_dict["c2_2"] = self.con_encoder(a2_t2) # content 2 in actor 2
        
        recons_dict["emos1_1"], recons_dict["e1_1"] = self.emo_encoder(a1_t1) # emotion 1 in actor 1
        recons_dict["emos2_1"], recons_dict["e2_1"] = self.emo_encoder(a2_t1) # emotion 1 in actor 2
        recons_dict["emos1_2"], recons_dict["e1_2"] = self.emo_encoder(a1_t2) # emotion 2 in actor 1
        recons_dict["emos2_2"], recons_dict["e2_2"] = self.emo_encoder(a2_t2) # emotion 2 in actor 2
        
        recons_dict["ids1_1"], recons_dict["s1_1"] = self.sty_encoder(a1_t1) # style 1 in actor 1
        recons_dict["ids2_1"], recons_dict["s2_1"] = self.sty_encoder(a2_t1) # style 1 in actor 2
        recons_dict["ids1_2"], recons_dict["s1_2"] = self.sty_encoder(a1_t2) # style 2 in actor 1
        recons_dict["ids2_2"], recons_dict["s2_2"] = self.sty_encoder(a2_t2) # style 2 in actor 2
        
        # self reconstruct
        recons_dict["self_a1_t1"] = self.decoder(recons_dict["c1_1"], recons_dict["e1_1"], recons_dict["s1_1"]) 
        # recons_dict["self_a1_t1"] = torch.permute(self_a1_t1, (0, 1, 3, 2))
        recons_dict["self_a1_t2"] = self.decoder(recons_dict["c1_2"], recons_dict["e1_2"], recons_dict["s1_2"])
        recons_dict["self_a2_t1"] = self.decoder(recons_dict["c2_1"], recons_dict["e2_1"], recons_dict["s2_1"])
        recons_dict["self_a2_t2"] = self.decoder(recons_dict["c2_2"], recons_dict["e2_2"], recons_dict["s2_2"])
        
        # content disentangle
        recons_dict["con_a1_t1"] = self.decoder(recons_dict["c2_1"], recons_dict["e1_1"], recons_dict["s1_1"])
        recons_dict["con_a2_t1"] = self.decoder(recons_dict["c1_1"], recons_dict["e2_1"], recons_dict["s2_1"])
        recons_dict["con_a1_t2"] = self.decoder(recons_dict["c2_2"], recons_dict["e1_2"], recons_dict["s1_2"])
        recons_dict["con_a2_t2"] = self.decoder(recons_dict["c1_2"], recons_dict["e2_2"], recons_dict["s2_2"])
        
        # emotion disentangle
        recons_dict["emo_a1_t1"] = self.decoder(recons_dict["c1_1"], recons_dict["e1_2"], recons_dict["s1_1"])
        recons_dict["emo_a2_t1"] = self.decoder(recons_dict["c2_1"], recons_dict["e2_2"], recons_dict["s2_1"])
        recons_dict["emo_a1_t2"] = self.decoder(recons_dict["c1_2"], recons_dict["e1_1"], recons_dict["s1_2"])
        recons_dict["emo_a2_t2"] = self.decoder(recons_dict["c2_2"], recons_dict["e2_1"], recons_dict["s2_2"])
        
        # style disentangle
        recons_dict["sty_a1_t1"] = self.decoder(recons_dict["c1_1"], recons_dict["e1_1"], recons_dict["s1_2"])
        recons_dict["sty_a2_t1"] = self.decoder(recons_dict["c2_1"], recons_dict["e2_1"], recons_dict["s2_2"])
        recons_dict["sty_a1_t2"] = self.decoder(recons_dict["c1_2"], recons_dict["e1_2"], recons_dict["s1_1"])
        recons_dict["sty_a2_t2"] = self.decoder(recons_dict["c2_2"], recons_dict["e2_2"], recons_dict["s2_1"])
        
        return recons_dict
            
    def accuracy(self, fake, label):
        _, pred = fake.topk(1, 1)
        pred0 = pred.squeeze().data
        acc = 100 * torch.sum(pred0 == label.data) / label.size(0)
        return acc
    
    def process(self, data, triplet):
        
        mfcc1_1 = Variable(data["a1_t1"].float())
        mfcc1_2 = Variable(data["a1_t2"].float())
        mfcc2_1 = Variable(data["a2_t1"].float())
        mfcc2_2 = Variable(data["a2_t2"].float())
        label = Variable(data["label"].long())
        label = torch.squeeze(label)
        ID1 = torch.squeeze(Variable(data["a1_ID"].long()))
        ID2 = torch.squeeze(Variable(data["a2_ID"].long()))
        reconstruct_dict = self.reconstruct(mfcc1_1, mfcc1_2, mfcc2_1, mfcc2_2)
        
        raise Exception(reconstruct_dict.keys())
        
        
        
        
        reconst_loss_map = {"self_a1_t1": data["a1_t1"], 
                            "self_a1_t2": data["a1_t2"], 
                            "self_a2_t1": data["a2_t1"], 
                            "self_a2_t2": data["a2_t2"],
                            "cross_a1_t1": data["a1_t1"],
                            "cross_a1_t2": data["a1_t2"],
                            "cross_a2_t1": data["a2_t1"],
                            "cross_a2_t2": data["a2_t2"]}
        
        # TODO: triplet implementation
        if triplet:
            raise NotImplementedError("Triplet loss not implemented yet")
        
        losses, accs = dict(), dict()
        for key in reconstruct_dict.keys():
            if key == "c1_1":
                losses["con_" + key] = self.l1_loss(reconstruct_dict[key], reconstruct_dict["c1_2"]) # 1 loss
            elif key == "c2_1":
                losses["con_" + key] = self.l1_loss(reconstruct_dict[key], reconstruct_dict["c2_2"]) # 1 loss
            elif key in ["e1_1", "e2_1", "e1_2", "e2_2"]:
                re = self.classify(reconstruct_dict[key])
                losses["es_" + key] = self.CroEn_loss(re, data["label"]) # 4 loss
                accs["es_" + key] = self.accuracy(re, data["label"]) # 4 acc
            elif key in ["c1_2", "c2_2"]:
                pass
            else:
                losses["rec_" + key] = self.l1_loss(reconstruct_dict[key], reconst_loss_map[key]) # 8 loss
        
        return reconstruct_dict, losses, accs    
    
    def update_learning_rate(self):
        pass
    
    def train_func(self, data, triplet=None):
        self.emo_encoder.train()
        self.decoder.train()
        self.con_encoder.train()
        self.sty_encoder.train()
        torch.set_grad_enabled(True)
        return self.process(data, triplet)
    
    def val_func(self, data, triplet=None):
        self.emo_encoder.eval()
        self.decoder.eval()
        self.con_encoder.eval()
        self.sty_encoder.eval()
        with torch.no_grad(): return self.process(data, triplet)
    
    def save_fig(self, data, reconstruct_dict, plot_path, epoch):
        # TODO: diff on heatmaps
        figures = {"Orig_a1_t1": data["a1_t1"], "Orig_a1_t2": data["a1_t2"], "Orig_a2_t1": data["a2_t1"], "Orig_a2_t2": data["a2_t2"],
                   "self_a1_t1": reconstruct_dict["self_a1_t1"], "self_a1_t2": reconstruct_dict["self_a1_t2"],
                   "self_a2_t1": reconstruct_dict["self_a2_t1"], "self_a2_t2": reconstruct_dict["self_a2_t2"],
                   "cross_a1_t1": reconstruct_dict["cross_a1_t1"], "cross_a1_t2": reconstruct_dict["cross_a1_t2"],
                   "cross_a2_t1": reconstruct_dict["cross_a2_t1"], "cross_a2_t2": reconstruct_dict["cross_a2_t2"]}
        
        _, axes = plt.subplots(3, 4, figsize=(20, 15))
        for i, (key, value) in enumerate(figures.items()):
            axes[i // 4, i % 4].set_title(key)
            g = value[0].squeeze()
            g = g.cpu().detach().numpy()
            sns.heatmap(g, ax=axes[i // 4, i % 4])
        plt.tight_layout()
        plt.savefig(plot_path + "/epoch_{}.png".format(epoch))

def _apply(layer, activation, normalizer, channel_out=None):
    if normalizer:
        layer.append(normalizer(channel_out))
    if activation:
        layer.append(activation())
    return layer

def conv2d(channel_in, channel_out,
           ksize=3, stride=1, padding=1,
           activation=nn.ReLU,
           normalizer=nn.BatchNorm2d):
    layer = list()
    bias = True if not normalizer else False

    layer.append(nn.Conv2d(channel_in, channel_out,
                     ksize, stride, padding,
                     bias=bias))
    _apply(layer, activation, normalizer, channel_out)
    # init.kaiming_normal(layer[0].weight)

    return nn.Sequential(*layer)

if __name__ == "__main__":
    
    model = AutoEncoder3x()
    
    data = dict()
    data["a1_t1"] = torch.randn(128, 1, 12, 28)
    data["a1_t2"] = torch.randn(128, 1, 12, 28)
    data["a2_t1"] = torch.randn(128, 1, 12, 28)
    data["a2_t2"] = torch.randn(128, 1, 12, 28)
    data["label"] = torch.randint(0, 8, (128,))
    data["a1_ID"] = torch.randint(0, 30, (128,))
    data["a2_ID"] = torch.randint(0, 30, (128,))
    
    reconstruct_dict, losses, accs = model.train_func(data, triplet=None)