
import torch
import textgrid
from transformers import GPT2Tokenizer, GPT2Model, GPT2TokenizerFast


def process_tg(dialogue, tokenizer, model, device="cpu"):
    
    encoded_input = tokenizer(dialogue, return_tensors='pt')
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    with torch.no_grad():
        output = model(**encoded_input)
    last_hidden_state = output[0]
    return last_hidden_state.squeeze(0) 

def process_tg_batch(dialogue_list, tokenizer, model, device="cpu"):
    
    encoded_input = tokenizer(dialogue_list, return_tensors='pt', padding=True)
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    with torch.no_grad():
        output = model(**encoded_input)
    last_hidden_state = output[0]
    return last_hidden_state # torch.Size([bs, seq_len, hidden])
        
    
if __name__ == "__main__":
    
    # Debug
    
    d_config = {
        "DATA_PARAM": {
            "Txtgrid": {
                "hf_model": "gpt2",
                "fps": 25
            }
        }
    }
    
    d_hf_model = d_config["DATA_PARAM"]["Txtgrid"]["hf_model"]
    # 'gpt2' 768, 'gpt2-medium' 1024, 'gpt2-large' 1280, 'gpt2-xl' 1600
    d_txt_tokenizer = GPT2Tokenizer.from_pretrained(d_hf_model)
    d_txt_tokenizer = GPT2TokenizerFast.from_pretrained(d_hf_model)
    d_txt_model = GPT2Model.from_pretrained(d_hf_model)
    
    from all_words import *
    all_corpus_dict = corpos_text
    i = 0
    for v in all_corpus_dict.values():
        i += 1
        debug_dialogue = v[0]
        # raise Exception("debug_dialogue", debug_dialogue)
        d_hidden_state = process_tg(debug_dialogue, d_txt_tokenizer, d_txt_model)
        print(i, d_hidden_state.shape, len(debug_dialogue.split())) # torch.Size([1, 200, 1280]) bs, seq_len, hidden_size
        # seq length ranges from 156 to 290 for 21 sentences