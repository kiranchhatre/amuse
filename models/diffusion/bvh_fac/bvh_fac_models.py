
import torch
import einops
import numpy as np
from torch import nn

from models.diffusion.utils.position_encoding import build_position_encoding
from models.diffusion.utils.cross_attention import (
    SkipTransformerEncoder,
    SkipTransformerDecoder,
    TransformerEncoderLayer,
    TransformerDecoderLayer
)
from models.diffusion.utils.temos_util import lengths_to_mask
from models.diffusion.utils.mdm_utils import (
    InputProcess,
    PositionalEncoding,
    EmbedOH,
    TimestepEmbedder,
    OutputProcess
)
from models.diffusion.utils.faceformer_utils import (
    PeriodicPositionalEncoding,
    enc_dec_mask,
    init_biased_mask
)
from models.diffusion.text.txt_util import TCNTxtNet
from models.diffusion.face.fac_util import FaceNet
from models.diffusion.audio.wav_models import AudioNet
from models.diffusion.utils.fusion_addon import FusionNet, AddonNet
from models.diffusion.discriminator.discriminator import ConvDiscriminator
        
class PoseMDM(nn.Module):
    
    def __init__(self, cfg, gpt_version, processed, bs, arch_type, modality):
        super().__init__()

        self.cfg = cfg
        self.processed = processed 
        self.arch_type = arch_type
        self.modality = modality
        self.bs = bs
        self.seq_len = self.cfg["seq_len"]
        self.arch = self.cfg["pose_encoder"]
        self.autoregressive = self.cfg["autoregressive"]
        self.data_rep = "" if self.arch == "lstm" else self.cfg[self.arch]["pose_rep"]
        self.arch_cfg = self.cfg[self.arch]
        self.latent_dim =  self.arch_cfg["latent_dim"]
        self.txt_latent_dim =  self.cfg["cond_mode"]["txt"]["latent_dim"]
        self.audio_latent_dim =  self.cfg["cond_mode"]["audio"]["latent_dim"]
        self.discriminator = self.arch_cfg["use_discriminator"]
        self.num_heads = self.arch_cfg["num_heads"]
        self.ff_size = self.arch_cfg["ff_size"]
        self.dropout = self.arch_cfg["dropout"]
        self.activation = self.arch_cfg["activation"]
        self.num_layers = self.arch_cfg["num_layers"]
        self.njoints = self.arch_cfg["njoints"][self.arch_type]
        self.nfeats = self.arch_cfg["nfeats"]
        self.b_f = self.arch_cfg["batch_first"]
        self.bi_dir = self.arch_cfg["bidir"]
        self.input_feats = self.njoints * self.nfeats
        self.gru_emb_dim = self.latent_dim if self.arch == "gru" else 0
        self.emb_trans_dec = self.cfg[self.arch]["emb_trans_dec"] if self.arch == "trans_dec" else None
        self.concat_emb_memory = self.cfg[self.arch]["concat_emb_memory"] if "concat_emb_memory" in self.arch_cfg else False
        self.gpt_version = gpt_version
        self.fusion = self.cfg["fusion"]["type"]
        self.fusion_latent_dim =  self.cfg["fusion"][self.fusion]["latent_dim"]
        self.use_diffusion = self.cfg["use_ddpm"]
        self.cond_mode = self.cfg["cond_mode"]["type"]                                                                      # CaMN main cond  | txt for ff/mdm
        self.cond_addon = self.cfg["cond_mode"]["addon"]                                                                    # CaMN addon cond | emo for ff
        self.cond_default_ff = self.cfg["cond_mode"]["default_ff"] if "default_ff" in self.cfg["cond_mode"] else None       # FF main cond    | --
        self.cond_default_mdm = self.cfg["cond_mode"]["default_mdm"] if "default_mdm" in self.cfg["cond_mode"] else None    # MDM main cond   | --
        self.cond_addon_mdm = self.cfg["cond_mode"]["addon_mdm"] if "addon_mdm" in self.cfg["cond_mode"] else None          # --              | emo/speaker for mdm
        self.force_mask = self.cfg["cond_mode"]["force_mask"] if "force_mask" in self.cfg["cond_mode"] else False           # masks for diff  | --
        self.audio_cond_mask_prob = self.cfg["cond_mode"]["audio"]["cond_mask_prob"] if "cond_mask_prob" in self.cfg["cond_mode"]["audio"] else 0.0
        self.txt_cond_mask_prob = self.cfg["cond_mode"]["txt"]["cond_mask_prob"] if "cond_mask_prob" in self.cfg["cond_mode"]["txt"] else 0.0
        assert self.audio_cond_mask_prob == self.txt_cond_mask_prob, "Audio and txt cond mask prob as kept same for now"
        
        self.face_arch = self.cfg["face_encoder"]
        self.face_cfg = self.cfg[self.face_arch]
        self.face_input_feats = self.face_cfg["njoints"][self.arch_type]
        self.face_latent_dim = self.face_cfg["latent_dim"]
        
        self.emotions =     {"netural": 0, "happiness": 1, "anger": 2, "sadness": 3,
                             "contempt": 4, "surprise": 5, "fear": 6, "disgust": 7}
        self.train_actors = {"yingqing": 0, "miranda": 1, "sophie": 2, "itoi": 3, "kieks": 4,
                             "luqi": 5, "carla": 6, "goto": 7, "li": 8, "reamey": 9, "jorge": 10,
                             "scott": 11, "katya": 12, "stewart": 13, "hailing": 14, "jaime": 15,
                             "lu": 16, "nidal": 17, "zhao": 18, "hanieh": 19, "carlos": 20, "lawrence": 21,
                             "ayana": 22, "daiki": 23}
        self.val_actors =   {"solomon": 24, "kexin": 25, "tiffnay": 26}
        self.test_actors =  {"catherine": 27, "zhang": 28, "wayne": 29}
        self.actor_ids =    {"train": self.train_actors, "val": self.val_actors, "test": self.test_actors}

        # CaMN based        
        if self.arch == "lstm": 
            
            self.fusion_in = 0
            self.fusion_string = ""
            if self.cond_mode != "unconstrained":
                if "txt" in self.cond_mode:
                    if self.arch_type == "content" or self.arch_type == "combo":
                        self.fusion_in += self.txt_latent_dim
                        self.fusion_string += "txt_"
                        self.txt_net = TCNTxtNet(self.cfg, self.gpt_version, processed=self.processed, bs=self.bs)
                    
                if "audio" in self.cond_mode:
                    self.fusion_in += self.audio_latent_dim
                    if "dis" in self.cond_mode:
                        self.fusion_string += "dis_audio_"
                        self.audio_net = AudioNet(self.cfg, nfeats=self.arch_type)
                    else:
                        self.fusion_string += "audio_"
                        self.audio_net = AudioNet(self.cfg, seq_len=self.seq_len)      # TODO: 
                        
                if self.cond_addon:    
                    if self.arch_type == "emotion" or self.arch_type == "combo":
                        if len(self.cond_addon.split("_")) > 1:
                            for addon_idx in range(len(self.cond_addon.split("_"))):
                                addon = self.cond_addon.split("_")[addon_idx]
                                self._prepare_addon_net(addon, self.fusion_in, self.fusion_string)
                        else:
                            self._prepare_addon_net(self.cond_addon, self.fusion_in, self.fusion_string)
                            
                self.fusion_net = FusionNet(self.cfg, self.fusion_in) 
            
            if self.modality == "pose":
                self.lstm_in = self.fusion_latent_dim + self.input_feats if self.fusion_in else self.input_feats
                self.lstm_latent_dim = self.latent_dim
                self.lstm_output_dim = self.input_feats
                
                if self.discriminator: # TODO add LSTM based discriminator
                    self.lstm_discriminator = ConvDiscriminator()

            elif self.modality == "face":
                self.facenet = FaceNet(self.face_cfg, self.arch_type)
                self.lstm_in = self.fusion_latent_dim + self.face_latent_dim if self.fusion_in else self.face_latent_dim
                self.lstm_latent_dim = self.face_latent_dim
                self.lstm_output_dim = self.face_input_feats
                self.b_f = self.face_cfg["batch_first"]
                self.bi_dir = self.face_cfg["bidir"]
                self.dropout = self.face_cfg["dropout"]
                self.num_layers = self.face_cfg["num_layers"]

            else: raise ValueError(f"Modality {self.modality} not supported, choose from ['pose', 'face']")
            
            self.lstm = nn.LSTM(self.lstm_in, hidden_size=self.lstm_latent_dim, 
                                num_layers=self.num_layers, batch_first=self.b_f,
                                bidirectional=self.bi_dir, dropout=self.dropout)
            self.lstm_final = nn.Sequential(
                nn.Linear(self.lstm_latent_dim, self.lstm_latent_dim//2),
                nn.LeakyReLU(True),
                nn.Linear(self.lstm_latent_dim//2, self.lstm_output_dim)
                )

        # MDM-Faceformer based   
        elif self.arch in ["trans_enc", "trans_dec", "gru", "trans_face_dec"]:
            
            # 1. pose model           
            if self.modality == "pose":
                
                # conditions: 
                if self.cond_mode != "unconstrained":
                    if 'txt' in self.cond_mode:
                        raise Exception(f"Text condition not supported for {self.arch} architecture")
                        # self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
                        # self.clip_version = clip_version
                        # self.clip_model = self.load_and_freeze_clip(clip_version)
                    else: raise Exception(f"Condition mode {self.cond_mode} not supported for {self.arch} architecture")
                assert "audio" in self.cond_default_mdm, "[MDM init] audio condition not specified"
                if "dis" not in self.cond_default_mdm:
                    if "combo_tcn_channels" not in self.cfg["cond_mode"]["audio"]:
                        self.audio_net = AudioNet(self.cfg, seq_len=self.seq_len)    
                    else:               
                        self.audio_net = AudioNet(self.cfg, seq_len=self.seq_len, tcn_channels=self.cfg["cond_mode"]["audio"]["combo_tcn_channels"])
                    audio_input_size = self.cfg["cond_mode"]["audio"]["combined_latent"] 
                else: audio_input_size = self.cfg["cond_mode"]["audio"]["dis"][self.arch_type]
                self.embed_audio = nn.Linear(audio_input_size, self.latent_dim)    
                if self.cond_addon_mdm is not None:
                    if "emotion" in self.cond_addon_mdm:
                        self.embed_emo = EmbedOH(len(self.emotions), self.latent_dim)
                    if "speaker" in self.cond_addon_mdm:
                        self.total_actors_count = len(self.train_actors) + len(self.val_actors) + len(self.test_actors)
                        self.embed_id = EmbedOH(self.total_actors_count, self.latent_dim)
                
                # networks
                self.input_process = InputProcess(self.data_rep, self.input_feats+self.gru_emb_dim, self.latent_dim)
                self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
                if self.arch == "trans_enc":
                    seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                                    nhead=self.num_heads,
                                                                    dim_feedforward=self.ff_size,
                                                                    dropout=self.dropout,
                                                                    activation=self.activation)
                    self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                                 num_layers=self.num_layers)
                elif self.arch == "trans_dec":
                    seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                                    nhead=self.num_heads,
                                                                    dim_feedforward=self.ff_size,
                                                                    dropout=self.dropout,
                                                                    activation=self.activation)
                    self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                                 num_layers=self.num_layers)
                elif self.arch == "gru":
                    self.gru = nn.GRU(self.latent_dim, self.latent_dim, num_layers=self.num_layers, batch_first=True)
                else: raise ValueError('Please choose correct architecture [trans_enc, trans_dec, gru]')
                self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
                self.output_process = OutputProcess(self.data_rep, self.input_feats, self.latent_dim, self.njoints, self.nfeats)
                
            # 2. face model    
            elif self.modality == "face":
                
                # conditions: audio, speaker, emotion TODO: txt
                assert "audio" in self.cond_default_ff, "[DIFF model init] Must provide audio default condition for MDM-Faceformer"
                assert "speaker" in self.cond_default_ff, "[DIFF model init] Must provide speaker default condition for MDM-Faceformer"
                if "dis" in self.cond_default_ff:
                    audio_input_size = self.cfg["cond_mode"]["audio"]["dis"][self.arch_type]
                else:
                    audio_input_size = self.cfg["cond_mode"]["audio"]["combined_latent"]   
                    self.audio_net = AudioNet(self.cfg, seq_len=self.seq_len)              # TODO: increase layers in TemporalConvNet() in txt_util.py 2800 -> 100
                self.audio_feature_map = nn.Linear(audio_input_size, self.face_latent_dim)
                self.total_actors_count = len(self.train_actors) + len(self.val_actors) + len(self.test_actors)
                self.obj_vector = nn.Linear(self.total_actors_count, self.face_latent_dim, bias=self.face_cfg["vec_bias"])
                if "txt" in self.cond_mode: raise Exception(f"Text condition not supported for {self.arch} architecture")
                if "emotion" in self.cond_addon:
                    self.emo_vector = nn.Linear(len(self.emotions), self.face_latent_dim, bias=self.face_cfg["vec_bias"])
                    
                # networks
                self.vertice_map = nn.Linear(self.face_input_feats, self.face_latent_dim)
                self.PPE = PeriodicPositionalEncoding(d_model=self.face_latent_dim, period=self.face_cfg["period"],
                                                      dropout=self.face_cfg["dropout"], max_seq_len=self.face_cfg["max_seq_len"])
                self.biased_mask = init_biased_mask(n_head=self.face_cfg["num_heads"], max_seq_len=self.face_cfg["max_seq_len"], period=self.face_cfg["period"])
                decoder_layer = nn.TransformerDecoderLayer(d_model=self.face_latent_dim, nhead=self.face_cfg["num_heads"], 
                                                           dim_feedforward=self.face_cfg["ff_size"], batch_first=self.face_cfg["batch_first"])        
                self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.face_cfg["num_layers"])
                self.vertice_map_r = nn.Linear(self.face_latent_dim, self.face_input_feats)
                nn.init.constant_(self.vertice_map_r.weight, 0)
                nn.init.constant_(self.vertice_map_r.bias, 0)
                
            else: raise ValueError(f"Modality {self.modality} not supported, choose from ['pose', 'face']")
            
        else: raise ValueError(f"Architecture {self.arch} not supported")
        
    def forward(self, x, mode):
        
        if self.arch == "lstm":
            """ 
            HACK: 
            1. upsampling for gpt2 embeddings using dynamic TCN when "txt" is used
            2. for all nets https://stackoverflow.com/questions/63465187/runtimeerror-cudnn-rnn-backward-can-only-be-called-in-training-mode

            Fusion information:
                content = txt + con_audio 
                emotion = emo_audio + emo_label +speaker_id
                combo = txt + raw_audio + emo_label + speaker_id
            """
            
            in_seq_len = x["pose_" + self.arch_type].shape[1]                   # all pose/ face sequences are of same length
            device = x["pose_" + self.arch_type].device
            assert in_seq_len == self.seq_len, "Input sequence length is not equal to the model sequence length"
            
            # Conditions
            fusion_feat, decoder_hidden = None, None
            if self.fusion_string:
                if "sid" in self.fusion_string:
                    actor_id_batch = []
                    for i in range(len(x["attr"][0])):
                        actor_id = self.actor_ids[mode][x["attr"][0][i]]
                        actor_id_batch.append(actor_id)
                    actor_id_batch = torch.tensor(actor_id_batch).to(device)
                    actor_id_batch = actor_id_batch[:, None]
                    speaker_feat = self.speaker_net(actor_id_batch)
                    speaker_feat = speaker_feat.repeat(1, in_seq_len, 1)
                    fusion_feat = speaker_feat if fusion_feat is None else torch.cat((fusion_feat, speaker_feat), dim=2)
                if "eid" in self.fusion_string:
                    emotion_id_batch = x["emo_label"].repeat(in_seq_len, 1).permute(1, 0)
                    emotion_feat = self.emotion_net(emotion_id_batch)
                    fusion_feat = emotion_feat if fusion_feat is None else torch.cat((fusion_feat, emotion_feat), dim=2)
                if "txt" in self.fusion_string:
                    txt_feat = self.txt_net(x["corpus_gpt"], x["attr"], device, in_seq_len, mode)
                    fusion_feat = txt_feat if fusion_feat is None else torch.cat((fusion_feat, txt_feat), dim=2)
                if "audio" in self.fusion_string:
                    if "dis" in self.fusion_string:
                        audio_feat = self.audio_net(audio_feats=x["wav_mfcc_" + self.arch_type])
                    else:
                        # audio_feat = self.audio_net(audio_feats=x["wav_mfcc"], dialogue_id=x["corpus_gpt"], attr=x["attr"], in_seq_len=in_seq_len, mode=mode) # obsolete
                        audio_feat = self.audio_net(audio_feats=x["wav_mfcc"])
                    fusion_feat = audio_feat if fusion_feat is None else torch.cat((fusion_feat, audio_feat), dim=2)
            
            # Pose        
            if self.modality == "pose":        
                if fusion_feat is not None:
                    prepare_fusion_feat = fusion_feat.reshape(-1, self.fusion_in)
                    fused_feat = self.fusion_net(prepare_fusion_feat)
                    fusion_feat = fused_feat.reshape(*audio_feat.shape) # TODO: change to robust shape for different condition: txt only etc
                    in_data = torch.cat((x["pre_pose_" + self.arch_type], fusion_feat), dim=2)
                else:
                    in_data = x["pre_pose_" + self.arch_type]

            # Face   
            elif self.modality == "face":
                if fusion_feat is not None:
                    prepare_fusion_feat = fusion_feat.reshape(-1, self.fusion_in)
                    fused_feat = self.fusion_net(prepare_fusion_feat)
                    fusion_feat = fused_feat.reshape(*audio_feat.shape) # TODO: change to robust shape for different condition: txt only etc
                    face_feat = self.facenet(x["pre_face_" + self.arch_type])
                    in_data = torch.cat((face_feat, fusion_feat), dim=2)
                else:
                    face_feat = self.facenet(x["pre_face_" + self.arch_type])
                    in_data = face_feat
                
            output, decoder_hidden = self.lstm(in_data, decoder_hidden)
            output = output[:, :, :self.latent_dim] + output[:, :, self.latent_dim:]
            output = self.lstm_final(output.reshape(-1, output.shape[2]))
            output = output.reshape(in_data.shape[0], in_data.shape[1], -1)
        
        elif self.arch in ["trans_enc", "trans_dec", "gru", "trans_face_dec"]:
            device = x["emo_label"].device
            
            # Pose        
            if self.modality == "pose":  
                """
                Fusion information:
                content = ts + txt + con_audio  
                emotion = ts + emo_audio + emo_label +speaker_id
                combo   = ts + txt + raw_audio + emo_label + speaker_id
                """ 
                if self.use_diffusion: emb = self.embed_timestep(x["ts"])   # [1, bs, d]
                actor_id = torch.tensor([self.actor_ids[mode][actor] for actor in x["attr"][0]]).to(device)
                if self.cond_addon_mdm is not None and self.arch_type in ["emotion", "combo"]:
                    if "emotion" in self.cond_addon_mdm: 
                        emo_feat = self.embed_emo(x["emo_label"])
                        emo_emb = self._mask_cond(emo_feat, force_mask=self.force_mask, cond_type="emo")
                        if self.use_diffusion: emb = emb + emo_emb
                    if "speaker" in self.cond_addon_mdm:
                        speaker_feat = self.embed_id(actor_id)
                        speaker_emb = self._mask_cond(speaker_feat, force_mask=self.force_mask, cond_type="speaker")
                        if self.use_diffusion: emb = emb + speaker_emb
                if 'txt' in self.cond_mode:
                    # enc_text = self._encode_text(y['text'])
                    # emb += self.embed_text(self._mask_cond(enc_text, force_mask=self.force_mask))
                    raise Exception(f"Text condition not supported for {self.arch} architecture")
                if "dis" in self.cond_default_mdm: audio_in = x["wav_mfcc_" + self.arch_type]
                else: audio_in = self.audio_net(audio_feats=x["wav_mfcc"], embproj=False)
                audio_feat = self.embed_audio(self._mask_cond(audio_in, force_mask=self.force_mask, cond_type="audio"))
                audio_feat = einops.rearrange(audio_feat, 'b t c -> t b c')
                
                if self.arch == 'trans_dec':
                    if self.use_diffusion:
                        noise = torch.randn_like(x["pose_" + self.arch_type])
                        x_t = x["base_diff"].q_sample(x_start=x["pose_" + self.arch_type], t=x["old_ts"], noise=noise)
                        x = self.input_process(x_t)
                        x = einops.rearrange(x, 'b t c -> t b c')
                        xseq = torch.cat((emb, x), axis=0)                                         
                        xseq = self.sequence_pos_encoder(xseq)                                      
                        if self.concat_emb_memory: memory = torch.cat((emb, audio_feat), axis=0)   
                        else: memory = audio_feat                                                   
                        output = self.seqTransDecoder(tgt=xseq, memory=memory)[1:]  
                    else:
                        x = self.input_process(x["pose_" + self.arch_type])
                        x = einops.rearrange(x, 'b t c -> t b c')
                        emb = 0
                        if self.arch_type in ["emotion", "combo"]: 
                            if "emotion" in self.cond_addon_mdm:
                                emb += emo_emb
                            if "speaker" in self.cond_addon_mdm:
                                emb += speaker_emb
                        if type(emb) == torch.Tensor: 
                            emb = emb.unsqueeze(0)
                            xseq = torch.cat((emb, x), axis=0)
                            xseq = self.sequence_pos_encoder(xseq)
                            if self.concat_emb_memory: memory = torch.cat((emb, audio_feat), axis=0)   
                            else: memory = audio_feat                                                   
                            output = self.seqTransDecoder(tgt=xseq, memory=memory)[1:] 
                        else:
                            xseq = self.sequence_pos_encoder(x)
                            memory = audio_feat
                            output = self.seqTransDecoder(tgt=xseq, memory=memory)
                elif self.arch == "trans_fftype_dec":
                    raise Exception(f"{self.arch} architecture not supported for {self.modality} modality")
                elif self.arch == 'trans_enc':
                    raise Exception(f"{self.arch} architecture not supported for {self.modality} modality")
                elif self.arch == 'gru':
                    raise Exception(f"{self.arch} architecture not supported for {self.modality} modality")
                else: raise Exception(f"Architecture {self.arch} not supported for {self.modality} modality")
                
                output = self.output_process(output)
                output = einops.rearrange(output, 't b c -> b t c')
            
            # Face   
            elif self.modality == "face":
                """
                Fusion information:
                content = speaker_id + con_audio + txt (upsupported)
                emotion = speaker_id + emo_audio + emo_label                                            
                combo = speaker_id + raw_audio + emo_label + txt (upsupported)               
                """
                
                # Transformer Enc Dec types
                # 1. Trans Enc: basic encoding + learning additional params (mu sig) + additional conditioning
                # 2. Trans Dec: combine two vecs tgt and hidden hidden_states
                # 3. Trans Enc and Dec both together
                # 4: VAE like: actor, teach, temos, Sinc  
                # 5. Training kind: autoregressive OR teacher forcing OR actor-like
                # 6. latent level: MLD
                # Masking stuff
                # https://stackoverflow.com/q/62170439
                # https://pytorch.org/tutorials/beginner/translation_transformer.html
                # https://nlp.seas.harvard.edu/2018/04/03/attention.html#positional-encoding
                # https://pytorch.org/docs/master/generated/torch.nn.MultiheadAttention.html#torch.nn.MultiheadAttention
                # https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
                # ways: concat xseq with cond passing to trans encoder (concat in the feat dim, to get (bs, seqlen, feat_motion + feat_cond)) OR passing cond as memory in trans decoder
                
                actor_id = [self.actor_ids[mode][actor] for actor in x["attr"][0]]
                actor_one_hot = torch.nn.functional.one_hot(torch.tensor(actor_id), num_classes=self.total_actors_count).to(dtype=torch.float32, device=device)
                obj_embedding = self.obj_vector(actor_one_hot)
                bs, frame_num, feats = x["face_" + self.arch_type].shape
                if "dis" in self.cond_default_ff: audio_feat = x["wav_mfcc_" + self.arch_type]
                else: audio_feat = self.audio_net(audio_feats=x["wav_mfcc"], embproj=False)
                if "emotion" in self.cond_addon:
                    emotion_one_hot = torch.nn.functional.one_hot(torch.tensor(x["emo_label"]), num_classes=len(self.emotions)).to(dtype=torch.float32, device=device)
                    emo_embedding = self.emo_vector(emotion_one_hot)
                    emo_embedding = emo_embedding.unsqueeze(1)
                hidden_states = self.audio_feature_map(audio_feat)
                
                if mode == "train":
                    if self.autoregressive:                                     # training: AR 
                        for i in range(frame_num): 
                            if i == 0:
                                vertice_emb = obj_embedding.unsqueeze(1)
                                if self.arch_type != "content":
                                    style_emb = vertice_emb + emo_embedding if "emotion" in self.cond_addon else vertice_emb
                                else: style_emb = vertice_emb
                                vertice_input = self.PPE(style_emb) 
                            else: vertice_input = self.PPE(vertice_emb)
                            tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(device=device)
                            memory_mask = enc_dec_mask(device, vertice_input.shape[1], hidden_states.shape[1])
                            vertice_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)
                            vertice_out = self.vertice_map_r(vertice_out)
                            new_output = self.vertice_map(vertice_out[:,-1,:]).unsqueeze(1)
                            new_output = new_output + style_emb
                            vertice_emb = torch.cat((vertice_emb, new_output), 1)
                    else:                                                       # training: teacher forcing 
                        vertice_emb = obj_embedding.unsqueeze(1)
                        style_emb = vertice_emb
                        start_token = torch.zeros((bs, 1, feats)).to(device=device)
                        vertice_input = torch.cat((start_token,x["face_" + self.arch_type][:,:-1,:]), 1) # shifted right
                        vertice_input = self.vertice_map(vertice_input)
                        if self.arch_type != "content":
                            vertice_input = vertice_input + style_emb + emo_embedding if "emotion" in self.cond_addon else vertice_input + style_emb
                        else: vertice_input = vertice_input + style_emb
                        vertice_input = self.PPE(vertice_input)
                        tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(device=device)
                        memory_mask = enc_dec_mask(device, vertice_input.shape[1], hidden_states.shape[1])
                        vertice_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)
                        vertice_out = self.vertice_map_r(vertice_out)
                
                else:                                                           # prediction: AR only
                    assert mode == "val", f"mode {mode} is not supported for face prediction"
                    # print(f"[BVH_FAC PRED] Face val mode using AR")
                    for i in range(frame_num): 
                        if i == 0:
                            vertice_emb = obj_embedding.unsqueeze(1)  
                            if self.arch_type != "content":
                                style_emb = vertice_emb + emo_embedding if "emotion" in self.cond_addon else vertice_emb
                            else: style_emb = vertice_emb
                            vertice_input = self.PPE(style_emb)                       
                        else: vertice_input = self.PPE(vertice_emb)
                        tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(device=device)
                        memory_mask = enc_dec_mask(device, vertice_input.shape[1], hidden_states.shape[1])
                        vertice_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)
                        vertice_out = self.vertice_map_r(vertice_out)
                        new_output = self.vertice_map(vertice_out[:,-1,:]).unsqueeze(1)
                        new_output = new_output + style_emb
                        vertice_emb = torch.cat((vertice_emb, new_output), 1)
                        
                output = vertice_out
                    
        else: raise NotImplementedError(f"Architecture {self.arch} is not implemented")
        return output      
    
    def _mask_cond(self, cond, force_mask=False, cond_type=None):
        bs = cond.shape[0]
        dims = len(cond.shape)
        if cond_type == "txt": cond_mask = self.txt_cond_mask_prob
        else: cond_mask = self.audio_cond_mask_prob                             # audio, emo, speaker
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and cond_mask > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * cond_mask).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            if dims == 3: return cond * (1. - mask[:, None])
            elif dims == 2: return cond * (1. - mask)
            else: raise Exception(f"cond shape {cond.shape} is not supported for masking")
        else:
            return cond
        
    def _prepare_addon_net(self, addon, fus_in, fus_str):
        if "emotion" in addon:
            self.emo_addon_latdim = self.cfg["cond_mode"][addon]["latent_dim"]
            self.fusion_in = fus_in + self.emo_addon_latdim
            self.fusion_string = fus_str + "eid_"
            self.emotion_net = AddonNet(self.cfg, "emotion")
        if "speaker" in addon:
            self.spe_addon_latdim = self.cfg["cond_mode"][addon]["latent_dim"]
            self.fusion_in = fus_in + self.spe_addon_latdim
            self.fusion_string = fus_str + "sid_"
            self.speaker_net = AddonNet(self.cfg, "speaker") 
    
    def _encode_text(self):
        # https://github.com/openai/CLIP
        raise NotImplementedError(f"Text encoding is not implemented for {self.arch}")
        # raw_text - list (batch_size length) of strings with input text prompts
        # device = next(self.parameters()).device
        # max_text_len = 20 if self.dataset in ['humanml', 'kit'] else None  # Specific hardcoding for humanml dataset
        # if max_text_len is not None:
        #     default_context_length = 77
        #     context_length = max_text_len + 2 # start_token + 20 + end_token
        #     assert context_length < default_context_length
        #     texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate
        #     # print('texts', texts.shape)
        #     zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
        #     texts = torch.cat([texts, zero_pad], dim=1)
        #     # print('texts after pad', texts.shape, texts)
        # else:
        #     texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate
        # return self.clip_model.encode_text(texts).float()
    
    def fid_calculator(self):
        raise NotImplementedError(f"FID calculation is not implemented for {self.arch}")
    
class PoseMLD(nn.Module):
    
    def __init__(self, cfg, nfeats, discriminator=False):
        
        super().__init__()
        
        self.input_feats = cfg["TRAIN_PARAM"]["diffusion"]["arch"]["pos_enc"]["nfeats"][nfeats]
        self.output_feats = cfg["TRAIN_PARAM"]["diffusion"]["arch"]["pos_enc"]["nfeats"][nfeats]
        self.arch = cfg["TRAIN_PARAM"]["diffusion"]["arch"]["pos_enc"]["type"]                               # all_encoder or encoder_decoder
        self.pe_type = cfg["TRAIN_PARAM"]["diffusion"]["arch"]["pos_enc"][self.arch]["pe_type"]              # mld
        self.position_embedding = cfg["TRAIN_PARAM"]["diffusion"]["arch"]["pos_enc"][self.arch]["pos_emb"]   # learned or sine    
        self.latent_size = cfg["TRAIN_PARAM"]["diffusion"]["arch"]["pos_enc"][self.arch]["latent_dim"][0]
        self.latent_dim = cfg["TRAIN_PARAM"]["diffusion"]["arch"]["pos_enc"][self.arch]["latent_dim"][1]
        self.num_heads = cfg["TRAIN_PARAM"]["diffusion"]["arch"]["pos_enc"][self.arch]["nhead"]
        self.num_layers = cfg["TRAIN_PARAM"]["diffusion"]["arch"]["pos_enc"][self.arch]["num_layers"]
        self.dropout = cfg["TRAIN_PARAM"]["diffusion"]["arch"]["pos_enc"][self.arch]["dropout"]
        self.ff_size = cfg["TRAIN_PARAM"]["diffusion"]["arch"]["pos_enc"][self.arch]["ff_size"]
        self.activation = cfg["TRAIN_PARAM"]["diffusion"]["arch"]["pos_enc"][self.arch]["activation"]
        self.normalize_before = cfg["TRAIN_PARAM"]["diffusion"]["arch"]["pos_enc"][self.arch]["normalize_before"]
        self.discriminator = discriminator
        
        if self.pe_type == "mld":
            self.query_pos_encoder = build_position_encoding(self.latent_dim, self.position_embedding)
            self.query_pos_decoder = build_position_encoding(self.latent_dim, self.position_embedding)
        else:
            raise ValueError(f"pe_type {self.pe_type} not supported")
        
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
        
        if self.arch == "all_encoder":
            decoder_norm = nn.LayerNorm(self.latent_dim)
            self.decoder = SkipTransformerEncoder(encoder_layer, self.num_layers, decoder_norm)
        elif self.arch == "encoder_decoder":
            decoder_layer = TransformerDecoderLayer(
                self.latent_dim,
                self.num_heads,
                self.ff_size,
                self.dropout,
                self.activation,
                self.normalize_before
            )
            decoder_norm = nn.LayerNorm(self.latent_dim)
            self.decoder = SkipTransformerDecoder(decoder_layer, self.num_layers, decoder_norm)
        else:
            raise ValueError(f"arch {self.arch} not supported")
        
        self.global_motion_token = nn.Parameter(torch.randn(self.latent_size * 2, self.latent_dim))
        self.skel_embedding = nn.Linear(self.input_feats, self.latent_dim)
        self.final_layer = nn.Linear(self.latent_dim, self.output_feats)
        
        def encode(self, features, lengths):
            """
            Args:
                features: 
                lengths: 
            Returns:
                latent: 
                dist: 
            """
            if lengths is None:
                lengths = [len(feature) for feature in features]
                
            bs, nframes, nfeats = features.shape
            mask = lengths_to_mask(lengths, device=features.device)
            
            x = self.skel_embedding(features)
            x = x.permute(1, 0, 2)
            dist = torch.tile(self.global_motion_token[:, None, :], (1, bs, 1))
            dist_masks = torch.ones((bs, dist.shape[0]), dtype=torch.bool, device=dist.device) # Check dist.device usage in global context
            aug_mask = torch.cat([dist_masks, mask], axis=1)
            x_seq = torch.cat((dist, x), axis=0)

            assert self.pe_type == "mld", f"pe_type {self.pe_type} not supported"
            x_seq = self.query_pos_encoder(x_seq)
            dist = self.encoder(x_seq, src_key_padding_mask=~aug_mask)[:dist.shape[0]]
                
            mu = dist[0:1, ...]
            logvar = dist[1:2, ...]
            
            std = logvar.exp().pow(0.5)
            dist = torch.distributions.Normal(mu, std)
            latent = dist.rsample()
            return latent, dist
        
        def decode(self, z, lengths):
            mask = lengths_to_mask(lengths, device=z.device)
            bs, nframes = mask.shape
            queries = torch.tile(nframes, bs, self.latent_dim, device=z.device)
            
            if self.arch == "all_encoder":
                xseq = torch.cat((z, queries), axis=0)
                z_mask = torch.ones((bs, self.latent_size), dtype=torch.bool, device=z.device)
                aug_mask = torch.cat([z_mask, mask], axis=1)

                assert self.pe_type == "mld", f"pe_type {self.pe_type} not supported"
                xseq = self.query_pos_decoder(xseq)
                output = self.decoder(xseq, src_key_padding_mask=~aug_mask)[z.shape[0]:]
                    
            elif self.arch == "encoder_decoder":
                assert self.pe_type == "mld", f"pe_type {self.pe_type} not supported"
                queries = self.query_pos_decoder(queries)
                output = self.decoder(tgt=queries, memory=z, tgt_key_padding_mask=~mask).squeeze(0)
                    
            output = self.final_layer(output)
            output[~mask.T] = 0
            features = output.permute(1, 0, 2)
            return features
        
        def get_mean_std(self, phase): # TODO
            if phase in ["train", "val"]:
                pass
            elif phase == "test":
                pass
            return # check motion_process.py
        
        def feat2pose(self, features):
            # TODO
            pass