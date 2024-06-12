
import torch
import numpy as np
from pathlib import Path

class Pretrained_AST_EVP():
    
    def __init__(self, device=None) -> None:
        if device is None: self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else: self.device = device
        
    def get_model(self, config, processed, tag, audio_ablation=None):
        
        assert any([audio_ablation in ["full", "identity", "emotion", "ast_baseline"], audio_ablation==None]), f"[LATDIFF] Invalid audio ablation flag: {audio_ablation}"
        
        self.tag = tag
        saved_model_path = processed.parents[1] / "saved-models" / config["TRAIN_PARAM"][tag]["pretrained_ast"]
        dtw_models = [f for f in saved_model_path.iterdir() if f.is_file() and "experiment_args.json" not in str(f)]
        self.frame_based_feats = config["TRAIN_PARAM"]["wav_dtw_mfcc"]["frame_based_feats"]
        print(f"[LATDIFF - INFER AST] Frame based feats flag: {self.frame_based_feats}")
        
        tEAcc, tPAcc = -np.inf, -np.inf
        best_dtw_model = None
        for dtw_model in dtw_models:
            model_id = Path(dtw_model).stem
            tea, tpa = self._get_num(model_id.split("_")[3]), self._get_num(model_id.split("_")[4])
            if audio_ablation in ["full", "ast_baseline", "emotion", "ast_baseline"] or audio_ablation==None:
                if tea > tEAcc: tEAcc, best_dtw_model = tea, dtw_model
            elif audio_ablation == "identity":
                if tpa > tPAcc: tPAcc, best_dtw_model = tpa, dtw_model
        chosen_epoch = self._get_num(Path(best_dtw_model).stem.split("_")[1])
        if int(chosen_epoch) == 0: best_dtw_model = [f for f in dtw_models if "_1_" in str(f)][0]
        print("[LATDIFF] (2/3) <===== Chosen AST model: ", best_dtw_model, " , loading state dict... =====>")
        
        from models import allmodels
        self.model = allmodels["wav_dtw_mfcc"]
        self.model = self.model.to(self.device)
        self.model.load_state_dict(torch.load(best_dtw_model))
        self.model.eval()
        
        return self.model
    
    def get_features(self, fbank):
        # raise error if gpu count is more than 1
        if torch.cuda.device_count() > 1: raise Exception("[LATDIFF] Multiple GPUs are trickey, memory leak troubleshoot pending")
        fbank = fbank.to(self.device)
        return self.model.eval_func(fbank, self.frame_based_feats)
    
    def get_reconstructed_fbank(self, fbank):
        if torch.cuda.device_count() > 1: raise Exception("[LATDIFF] Multiple GPUs are trickey, memory leak troubleshoot pending")
        fbank = fbank.to(self.device)
        return self.model.eval_func(fbank, self.frame_based_feats, metrics=True)

    def _get_num(self, x):
        chars = [ele if ele.isdigit() or ele == '.' else ' ' for ele in x]
        num_str = ''.join(chars).split()
        if num_str:
            return float(num_str[0])
        return None
