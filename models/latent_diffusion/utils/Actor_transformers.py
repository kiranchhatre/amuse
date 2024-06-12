import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


# only for ablation / not used in the final model
class TimeEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(TimeEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask, lengths):
        time = mask * 1/(lengths[..., None]-1)
        time = time[:, None] * torch.arange(time.shape[1], device=x.device)[None, :]
        time = time[:, 0].T
        # add the time encoding
        x = x + time[..., None]
        return self.dropout(x)
    

class Encoder_TRANSFORMER(nn.Module):
    def __init__(self, njoints=47, nfeats=6, num_frames=300, num_classes=8, # num_classes=8, num_classes=30,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 ablation="average_encoder", activation="gelu", **kargs):
        super().__init__()
        
        # self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_frames = num_frames
        self.num_classes = num_classes
        
        # self.pose_rep = pose_rep
        # self.glob = glob
        # self.glob_rot = glob_rot
        # self.translation = translation
        
        self.latent_dim = latent_dim
        
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation
        
        self.input_feats = self.njoints*self.nfeats
        
        if self.ablation == "average_encoder":
            self.mu_layer = nn.Linear(self.latent_dim, self.latent_dim)
            self.sigma_layer = nn.Linear(self.latent_dim, self.latent_dim)
        else:
            self.muQuery = nn.Parameter(torch.randn(self.num_classes, self.latent_dim))
            # self.sigmaQuery = nn.Parameter(torch.randn(self.num_classes, self.latent_dim))
        
        self.skelEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        
        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        
        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=self.activation)
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                     num_layers=self.num_layers)

    def forward(self, batch):
        x, y, mask = batch["x"], batch["y"], batch["mask"]
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats)
        
        # embedding of the skeleton
        x = self.skelEmbedding(x)

        # only for ablation / not used in the final model
        if self.ablation == "average_encoder":
            # add positional encoding
            x = self.sequence_pos_encoder(x)
            
            # transformer layers
            final = self.seqTransEncoder(x, src_key_padding_mask=~mask)
            # get the average of the output
            z = final.mean(axis=0)
            
            # extract mu and logvar
            mu = self.mu_layer(z)
            logvar = self.sigma_layer(z)
        else:
            # adding the mu and sigma queries
            # xseq = torch.cat((self.muQuery[y][None], self.sigmaQuery[y][None], x), axis=0)
            xseq = torch.cat((self.muQuery[y][None], x), axis=0)

            # add positional encoding
            xseq = self.sequence_pos_encoder(xseq)

            # create a bigger mask, to allow attend to mu and sigma
            mu_only_Mask = torch.ones((bs, 1), dtype=bool, device=x.device)
            maskseq = torch.cat((mu_only_Mask, mask), axis=1)
            # muandsigmaMask = torch.ones((bs, 2), dtype=bool, device=x.device)
            # maskseq = torch.cat((muandsigmaMask, mask), axis=1)

            final = self.seqTransEncoder(xseq, src_key_padding_mask=~maskseq)
            mu = final[0]
            # logvar = final[1]
            
        # return {"mu": mu, "logvar": logvar}
        return {"mu": mu}


class Decoder_TRANSFORMER(nn.Module):
    def __init__(self, njoints=47, nfeats=6, num_frames=300, num_classes=8, # num_classes=8, num_classes=30,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1, activation="gelu",
                 ablation=None, **kargs):
        super().__init__()

        # self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_frames = num_frames
        self.num_classes = num_classes
        
        # self.pose_rep = pose_rep
        # self.glob = glob
        # self.glob_rot = glob_rot
        # self.translation = translation
        
        self.latent_dim = latent_dim
        
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation

        self.activation = activation
                
        self.input_feats = self.njoints*self.nfeats

        # only for ablation / not used in the final model
        if self.ablation == "zandtime":
            self.ztimelinear = nn.Linear(self.latent_dim + self.num_classes, self.latent_dim)
        else:
            self.actionBiases = nn.Parameter(torch.randn(self.num_classes, self.latent_dim))

        # only for ablation / not used in the final model
        if self.ablation == "time_encoding":
            self.sequence_pos_encoder = TimeEncoding(self.dropout)
        else:
            self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        
        seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=activation)
        self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                     num_layers=self.num_layers)
        
        self.finallayer = nn.Linear(self.latent_dim, self.input_feats)
        
    def forward(self, batch):
        z, y, mask, lengths = batch["z"], batch["y"], batch["mask"], batch["lengths"]

        latent_dim = z.shape[1]
        bs, nframes = mask.shape
        njoints, nfeats = self.njoints, self.nfeats

        # only for ablation / not used in the final model
        if self.ablation == "zandtime":
            yoh = F.one_hot(y, self.num_classes)
            z = torch.cat((z, yoh), axis=1)
            z = self.ztimelinear(z)
            z = z[None]  # sequence of size 1
        else:
            # only for ablation / not used in the final model
            if self.ablation == "concat_bias":
                # sequence of size 2
                z = torch.stack((z, self.actionBiases[y]), axis=0)
            else:
                # shift the latent noise vector to be the action noise
                # z = z + self.actionBiases[y] # AMUSE: completely removing the action label information
                z = z[None]  # sequence of size 1   
            
        timequeries = torch.zeros(nframes, bs, latent_dim, device=z.device)
        
        # only for ablation / not used in the final model
        if self.ablation == "time_encoding":
            timequeries = self.sequence_pos_encoder(timequeries, mask, lengths)
        else:
            timequeries = self.sequence_pos_encoder(timequeries)
        
        output = self.seqTransDecoder(tgt=timequeries, memory=z,
                                      tgt_key_padding_mask=~mask)
        
        output = self.finallayer(output).reshape(nframes, bs, njoints, nfeats)
        
        # zero for padded area
        output[~mask.T] = 0
        output = output.permute(1, 2, 3, 0)
        
        batch["output"] = output
        return batch
    
class ACTOR_AE(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()

        self.encoder = Encoder_TRANSFORMER()
        # self.decoder = Decoder_TRANSFORMER()
        
        # crossentropy layers
        self.crossentropy = nn.CrossEntropyLoss()
        self.classification_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 8) # EMOTION CLASSES
            # nn.Linear(64, 30) # PERSONALITY CLASSES
        )
        
    def forward(self, x, emo_label):
        batch = dict()
        batch["x"] = x
        
        y = emo_label
        lengths = torch.tensor([batch["x"].shape[-1]] * batch["x"].shape[0]).to(batch["x"].device)
        mask = torch.ones((batch["x"].shape[0], batch["x"].shape[-1]), dtype=bool).to(batch["x"].device)
        
        batch["mask"], batch["y"] = mask, y
        batch = self.encoder(batch)
        z = self.reparameterize(batch, vae=False)
        batch["y"], batch["mask"], batch["lengths"], batch["z"] = y, mask, lengths, z
        # batch = self.decoder(batch)
        
        predicted_labels = self.classification_head(batch["z"])
        prediction = F.softmax(predicted_labels, dim=1)
        
        # New
        batch["output"] = None
        loss = self.compute_loss(x, batch["output"], emo_label, predicted_labels)
        
        return {"output": batch["output"], "recon_loss": loss["recon_loss"], "class_loss": loss["class_loss"], 
                "predicted_labels": predicted_labels, "prediction": prediction, "z": z}
    
    def compute_loss(self, x, output, emo_label, predicted_labels):
        
        # # reconstruction loss
        # recon_loss = F.mse_loss(x, output)
        
        # classification loss
        emo_label = emo_label.long()
        class_loss = self.crossentropy(predicted_labels, emo_label)
        
        # return {"recon_loss": recon_loss, "class_loss": class_loss}
        return {"recon_loss": torch.tensor(0.0).to(x.device), "class_loss": class_loss}
    
    def reparameterize(self, batch, seed=None, vae=False):
        
        if vae:
            mu, logvar = batch["mu"], batch["logvar"]
            std = torch.exp(logvar / 2)

            if seed is None:
                eps = std.data.new(std.size()).normal_()
            else:
                generator = torch.Generator(device=mu.device)
                generator.manual_seed(seed)
                eps = std.data.new(std.size()).normal_(generator=generator)

            z = eps.mul(std).add_(mu)
            raise Exception("Not part of the final model!")
            return z
        else:
            # mu, logvar = batch["mu"], batch["logvar"]
            mu = batch["mu"]
            z = mu
            return z 
    
class ACTOR_AE_old(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()

        self.encoder = Encoder_TRANSFORMER()
        self.decoder = Decoder_TRANSFORMER()
        
        # crossentropy layers
        self.crossentropy = nn.CrossEntropyLoss()
        self.classification_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            # nn.Linear(64, 8) # EMOTION CLASSES
            nn.Linear(64, 30) # PERSONALITY CLASSES
        )
        
    def forward(self, x, emo_label):
        batch = dict()
        batch["x"] = x
        
        y = emo_label
        lengths = torch.tensor([batch["x"].shape[-1]] * batch["x"].shape[0]).to(batch["x"].device)
        mask = torch.ones((batch["x"].shape[0], batch["x"].shape[-1]), dtype=bool).to(batch["x"].device)
        
        batch["mask"], batch["y"] = mask, y
        batch = self.encoder(batch)
        z = self.reparameterize(batch, vae=False)
        batch["y"], batch["mask"], batch["lengths"], batch["z"] = y, mask, lengths, z
        batch = self.decoder(batch)
        
        predicted_labels = self.classification_head(batch["z"])
        prediction = F.softmax(predicted_labels, dim=1)
        
        loss = self.compute_loss(x, batch["output"], emo_label, predicted_labels)
        
        return {"output": batch["output"], "recon_loss": loss["recon_loss"], "class_loss": loss["class_loss"], 
                "predicted_labels": predicted_labels, "prediction": prediction, "z": z}
    
    def compute_loss(self, x, output, emo_label, predicted_labels):
        
        # reconstruction loss
        recon_loss = F.mse_loss(x, output)
        
        # classification loss
        emo_label = emo_label.long()
        class_loss = self.crossentropy(predicted_labels, emo_label)
        
        return {"recon_loss": recon_loss, "class_loss": class_loss}

    def reparameterize(self, batch, seed=None, vae=False):
        
        if vae:
            mu, logvar = batch["mu"], batch["logvar"]
            std = torch.exp(logvar / 2)

            if seed is None:
                eps = std.data.new(std.size()).normal_()
            else:
                generator = torch.Generator(device=mu.device)
                generator.manual_seed(seed)
                eps = std.data.new(std.size()).normal_(generator=generator)

            z = eps.mul(std).add_(mu)
            raise Exception("Not part of the final model!")
            return z
        else:
            # mu, logvar = batch["mu"], batch["logvar"]
            mu = batch["mu"]
            z = mu
            return z 

if __name__ == "__main__":
    
    # x torch.Size([20, 25, 6, 60]) # batch_size:20, joints:25, rot6, frames:60 ==> fps:30
    
    def reparam(batch, seed=None, device="cpu", vae=False):
        
        if vae:
            mu, logvar = batch["mu"], batch["logvar"]
            std = torch.exp(logvar / 2)

            if seed is None:
                eps = std.data.new(std.size()).normal_()
            else:
                generator = torch.Generator(device=device)
                generator.manual_seed(seed)
                eps = std.data.new(std.size()).normal_(generator=generator)

            z = eps.mul(std).add_(mu)
            return z
        else:
            mu, logvar = batch["mu"], batch["logvar"]
            z = mu
            return z 
    
    # enc = Encoder_TRANSFORMER(modeltype="cvae", njoints=25, nfeats=6, num_frames=60, num_classes=2, translation=True,
    #                             pose_rep="xyz", glob=True, glob_rot=True, latent_dim=256, ff_size=1024, num_layers=4,
    #                             num_heads=4, dropout=0.1, ablation=None, activation="gelu")
    # dec = Decoder_TRANSFORMER(modeltype="cvae", njoints=25, nfeats=6, num_frames=60, num_classes=2, translation=True,
    #                             pose_rep="xyz", glob=True, glob_rot=True, latent_dim=256, ff_size=1024, num_layers=4,
    #                             num_heads=4, dropout=0.1, ablation=None, activation="gelu")
    # x = torch.randn(20, 25, 6, 60)
    # y = torch.randint(0, 2, (20,))
    # mask = torch.ones((20, 60), dtype=bool)
    # lengths = torch.randint(1, 60, (20,))
    
    # batch = {"x": x, "y": y, "mask": mask, "lengths": lengths}
    # batch = enc(batch)
    # batch["z"] = reparam(batch, device="cpu", vae=False)
    # batch["y"] = torch.randint(0, 2, (20,))
    # batch["mask"] = mask
    # batch["lengths"] = lengths
    # batch = dec(batch)
    
    # print(batch["output"].shape)
    
    # enc = Encoder_TRANSFORMER()
    # dec = Decoder_TRANSFORMER()
    # x = torch.randn(64, 47, 6, 300)
    # y = torch.randint(0, 8, (64,))
    # mask = torch.ones((64, 300), dtype=bool)
    # lengths = torch.randint(1, 300, (64,))
    
    # batch = {"x": x, "y": y, "mask": mask, "lengths": lengths}
    # batch = enc(batch)
    # batch["z"] = reparam(batch, device="cpu", vae=False)
    # batch["y"] = y
    # batch["mask"] = mask
    # batch["lengths"] = lengths
    # batch = dec(batch)
    
    # print(batch["output"].shape) # torch.Size([64, 47, 6, 300])
    
    actor_ae = ACTOR_AE()
    x = torch.randn(64, 47, 6, 300) # bs, njoints, nfeats, nframes = x.shape
    y = torch.randint(0, 8, (64,))
    mask = torch.ones((64, 300), dtype=bool)
    lengths = torch.randint(1, 300, (64,))
    batch = {"x": x, "y": y, "mask": mask, "lengths": lengths}
    batch = actor_ae(batch)
    print(batch["output"].shape) # torch.Size([64, 47, 6, 300])
  
    
     
   
    