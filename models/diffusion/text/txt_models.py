
import json
import torch
import pickle
import textgrid
from torch import nn
from pathlib import Path
from transformers import GPT2Tokenizer, GPT2Model, GPT2TokenizerFast

from dm.utils.all_words import *
from dm.utils.corpus_utils import *

class TxtNet(nn.Module):
    # TODO: compare performance of emb_proj against TCN
    def __init__(self, cfg, gpt_version, processed, bs) -> None:
        super().__init__()
        
        self.bs = bs
        self.nonlinear_proj = cfg["cond_mode"]["txt"]["nonlinear"]
        self.input_size = cfg["cond_mode"]["txt"]["input_size"][gpt_version]
        self.hidden_size = cfg["cond_mode"]["txt"]["latent_dim"]
        self.all_corpus_dict = corpos_text # imported from dm/utils/all_words.py
        
        self.processed = processed
        with open(str(Path(self.processed, "eng_data_processed/all_eng_extracted_data.pkl")), "rb") as f:
            self.all_data = pickle.load(f)
        
        self.tokenizer = cfg["cond_mode"]["txt"]["tokenizer"]
        if self.tokenizer == "fast":
            self.txt_tokenizer = GPT2TokenizerFast.from_pretrained(gpt_version)
            self.txt_tokenizer.pad_token = self.txt_tokenizer.eos_token
        elif self.tokenizer == "gpt2":
            self.txt_tokenizer = GPT2Tokenizer.from_pretrained(gpt_version)
            self.txt_tokenizer.pad_token = self.txt_tokenizer.eos_token
        else: raise Exception(f"[DIFF - TXT MODEL] Unknown tokenizer: {self.tokenizer} [gpt2, fast]")
        self.txt_model = GPT2Model.from_pretrained(gpt_version)
        self.txt_model.eval() # set GPT model to evaluation mode
        for param in self.txt_model.parameters():
            param.requires_grad = False
        
        if self.nonlinear_proj:
            self.emb_proj = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.input_size, self.hidden_size)
            )
        else:
            self.emb_proj = nn.Linear(self.input_size, self.hidden_size)
        
    def forward(self, dialogue_id, attr, device):
        
        # # Full dialogue processing (Incorrect due pauses in the dialogue)
        # take = "_".join(dialogue_id.split("_")[2:]) # dialogue_id = "4_18_0_10_10"
        # full_dialogue = self.all_corpus_dict[take][0]
        # words = full_dialogue.split(" ")
        # words_per_div = len(words) // int(total_divs)
        # dialogue_cut_old = " ".join(words[int(index) * words_per_div : (int(index) + 1) * words_per_div])
        
        in_bs = len(dialogue_id[0])
        assert in_bs == self.bs, "[DIFF - TXT MODEL] dataloader batch size should be equal to set batch size"
        
        dialogue_cuts = [] 
        for i in range(len(dialogue_id[0])):
            d_id = dialogue_id[0][i]
            index = d_id.split("_")[0]
            total_divs = d_id.split("_")[1]
            assert int(index) < int(total_divs), "[DIFF - TXT MODEL] index should be less than total_divs"
            
            # Txtgrid processing
            actor = attr[0][i]
            take = "_".join(d_id.split("_")[2:])
            txt_file = self.all_data[actor][take]["txt"][0]
            tg = textgrid.TextGrid.fromFile(txt_file)
            time_per_div = tg[0][-1].maxTime / int(total_divs)
            min_time = time_per_div * int(index)
            max_time = time_per_div * (int(index) + 1)

            dialogue_cut = ""
            for word in tg[0]:
                if word.minTime >= min_time and word.maxTime <= max_time:
                    dialogue_cut += word.mark + " "
            dialogue_cuts.append(dialogue_cut)
        
        last_hidden_state = process_tg_batch(dialogue_cuts, self.txt_tokenizer, self.txt_model, device)
        txt_emb_latent = self.emb_proj(last_hidden_state) 
        return txt_emb_latent

