
import json
import wandb
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

from models.diffusion.diffusion_losses import get_loss_fn

class EvaluatorWrapper(object):
    # Wrapper for evaluations: R-precision, Multimodal-Dist, FID, Diversity, Multimodality
    pass

class GeneratedDataset(Dataset):

    def __init__(self, model, diffusion, dataloader, mm_num_repeats, mm_num_samples, num_samples_limit, 
                 dis_representation, diff_cfg, repeat_times, visualizer, model_path, viz_dict, eval_type):
        
        eval_type_r = ""
        if eval_type == "demo":
            eval_type_r = eval_type
            eval_type = "val"
            loader_count = 1
            batch_start = torch.randint(0, len(dataloader), (loader_count,)).item() # one random val batch
            batch_end = batch_start + 1
            viz_iter = batch_start
        elif eval_type == "train":
            loader_count = 1
            batch_start = torch.randint(0, len(dataloader), (loader_count,)).item() # one random train batch
            batch_end = batch_start + 1
            viz_iter = batch_start
        elif eval_type == "val":
            batch_start = 0
            batch_end = len(dataloader)
            viz_iter = torch.randint(0, len(dataloader), (1,)).item()               # full val set
                    
        # visualization    
        if not eval_type_r == "demo":
            viz_path = model_path / "viz" / eval_type
            viz_path.mkdir(parents=True, exist_ok=True)
        else:
            viz_path = model_path.parents[1] / "viz_dump" / "diffusion_" / "demo"
            viz_path.mkdir(parents=True, exist_ok=True)
        epoch = viz_dict["epoch"]
        bvh_version = viz_dict["bvh_version"]
        norm_bvh = viz_dict["norm_bvh"]
        debug = viz_dict["debug"]                                               # overrun by base_debug
        dwnsmpl_fac = viz_dict["dwnsmpl_fac"]
        bvh_fps = viz_dict["bvh_fps"]
        modality = viz_dict["modality"]
        
        with open(str(model_path.parents[1] / "configs/base.json"), "r") as f: base_cfg = json.load(f)
        base_debug = base_cfg["TRAIN_PARAM"]["debug"]
        
        clip_denoised = False
        skip_timesteps = 0
        
        GT_motion = []
        repeated_seq = []
        generated_motion = []
        generated_viz_motion = []
        mm_generated_motions = []                                               # TODO: multimodal
        
        real_num_batches = len(dataloader)
        if num_samples_limit is not None:
            real_num_batches = num_samples_limit // dataloader.batch_size + 1
        print(f"[GeneratedDataset] real_num_batches: {real_num_batches}")
        
        if mm_num_samples > 0:
            mm_idxs = np.random.choice(real_num_batches, mm_num_samples // dataloader.batch_size +1, replace=False)
            mm_idxs = np.sort(mm_idxs)
        else:
            mm_idxs = []
        print(f"[GeneratedDataset] mm_idxs: {mm_idxs}")
        
        loss_pose = diff_cfg["loss_pose"]
        for loss in loss_pose:
            if "lambda_l2" in loss:
                self.lambda_l2_loss = get_loss_fn(diff_cfg["loss_factory"][loss])
            
        sample_fn = diffusion.p_sample_loop
        model.eval()
        val_iters = 0
        generated_video = ""
        with torch.no_grad():
            for i, batch in list(enumerate(dataloader))[batch_start:batch_end]:
                losses = {}
                val_iters += 1
                if dis_representation:
                    all_motion   = {"pose_content": batch["pose_content"], "pose_emotion": batch["pose_emotion"]}
                    model_kwargs = {"wav_mfcc_content": batch["wav_mfcc_content"], "wav_mfcc_emotion": batch["wav_mfcc_emotion"], 
                                    "attr": batch["attr"], "corpus_gpt": batch["corpus_gpt"], "emo_label": batch["emo_label"], "mode": eval_type}
                else:
                    all_motion   = {"pose_combo": batch["pose_combo"]}
                    model_kwargs = {"wav_mfcc": batch["wav_mfcc"], "attr": batch["attr"], "corpus_gpt": batch["corpus_gpt"], 
                                    "emo_label": batch["emo_label"], "mode": eval_type}
                
                # is_mm = i in mm_idxs                                          # TODO: multimodal
                # repeat_times = mm_num_repeats if is_mm else 1
                
                # repetitions
                local_repeat_times = repeat_times if i == viz_iter else 1
                repeated_seq.append(0) if local_repeat_times == 1 else repeated_seq.extend([1] * local_repeat_times)
                print(f"[GeneratedDataset] eval: {eval_type} {i+1}/{len(dataloader)}, loader: {batch_start}/{batch_end}, viz iter: {viz_iter + 1}, batch reps: {local_repeat_times}")
                for _ in range(local_repeat_times):
                    
                    sample = sample_fn(
                        model,
                        all_motion,
                        # motion.shape,                                         # Obsolete
                        clip_denoised=clip_denoised,
                        model_kwargs=model_kwargs, 
                        skip_timesteps=skip_timesteps, 
                        init_image=None,
                        progress=True,                                          # True for diffusion infer progress bar else False
                        dump_steps=None,
                        noise=None,
                        const_noise=False,
                    )
                    if local_repeat_times != 1: generated_viz_motion.append(sample)
                    generated_motion.append(sample)
                    GT_motion.append(all_motion)

                if eval_type == "val":
                    device = sample["pose_content"].device if dis_representation else sample["pose_combo"].device
                    if dis_representation:
                        truncated_batch = {k: v for k, v in batch.items() if k in ["pose_content", "pose_emotion"]}
                        for k, v in truncated_batch.items(): truncated_batch[k] = v.to(device)
                        loss, rec_wt = self.lambda_l2_loss(sample, truncated_batch)
                        for k, v in loss.items():
                            if k == "pose_content": losses["pose_content"] = v.mean() * rec_wt
                            elif k == "pose_emotion": losses["pose_emotion"] = v.mean() * rec_wt
                            else: raise TypeError(f"Loss type {k} not supported.")
                    else:
                        truncated_batch = {k: v for k, v in batch.items() if k in ["pose_combo"]}
                        for k, v in truncated_batch.items(): truncated_batch[k] = v.to(device)
                        loss, rec_wt = self.lambda_l2_loss(sample, truncated_batch)
                        for k, v in loss.items():
                            if k == "pose_combo": losses["pose_combo"] = v.mean() * rec_wt
                            else: raise TypeError(f"Loss type {k} not supported.")
                    total_pose_loss = sum({k: v for k, v in losses.items() if k.startswith("pose_")}.values())
                    
                    if val_iters % 8 == 0:
                        if dis_representation: print(f"[GeneratedDataset - {eval_type}] con loss: {losses['pose_content']:.4f}, emo loss: {losses['pose_emotion']:.4f}, total loss: {total_pose_loss:.4f}")
                        else: print(f"[GeneratedDataset - {eval_type}] total loss: {total_pose_loss:.4f}")
                
                if i == viz_iter:
                    assert len(generated_viz_motion) == repeat_times, f"[GeneratedDataset] viz motion length: {len(generated_viz_motion)} != repeat times: {repeat_times}"
                    generated_video = visualizer.animate_sample(generated_viz_motion, batch, viz_path, epoch, bvh_version, norm_bvh, 
                                                                dwnsmpl_fac, bvh_fps, base_debug, mode=eval_type, modality=modality, diff_eval=True)
            
            if eval_type_r != "demo":
                assert generated_video != "", "[GeneratedDataset] generated video not found."
                print(f"[GeneratedDataset - {eval_type}] generated video: {generated_video}")
                if not base_debug:
                    wandb.log({f"diffeval_{eval_type}_video": wandb.Video(str(generated_video), fps=bvh_fps, format="mp4"), "epoch": epoch})
                    if eval_type == "val":
                        for k, v in losses.items():
                            wandb.log({f"diffeval_{eval_type}_{k}": v.item(), "epoch": epoch})
                        wandb.log({f"diffeval_{eval_type}_total_pose_loss": total_pose_loss, "epoch": epoch})
            
            else:
                assert eval_type_r == "demo", "[GeneratedDataset] eval_type_r should be demo."
                print(f"[GeneratedDataset - {eval_type}] generated video: {generated_video}")
          
        assert len(generated_motion) == len(GT_motion) == len(repeated_seq), f"[GeneratedDataset] generated_motion: {len(generated_motion)}, GT_motion: {len(GT_motion)}, repeated_seq: {len(repeated_seq)}"
        self.GT_motion = GT_motion
        self.generated_motion = generated_motion
        self.repeated_seq = repeated_seq

    def __len__(self):
        return len(self.generated_motion)

    def __getitem__(self, index):
        generated = self.generated_motion[index]
        GT = self.GT_motion[index]
        repeated_seq = self.repeated_seq[index]
        return {"GT": GT, "generated": generated, "repeated_seq": repeated_seq}
    
class DiffusionEval():
    
    def __init__(self, model, diffusion, dataloader, mm_num_repeats, mm_num_samples, num_samples_limit, 
                 dis_representation, diff_cfg, replication_times, visualizer, model_path, viz_dict, eval_type):
        self.model = model
        self.diffusion = diffusion
        self.dataloader = dataloader
        self.eval_type = eval_type
        self.dis_representation = dis_representation
        self.diff_cfg = diff_cfg
        self.replication_times = replication_times
        self.visualizer = visualizer
        self.model_path = model_path
        self.viz_dict = viz_dict
        self.mm_num_repeats = mm_num_repeats
        self.mm_num_samples = mm_num_samples
        self.num_samples_limit = num_samples_limit
        self.motion_loader = self._get_eval_loader()
        
    def _get_eval_loader(self):                                                 
        dataset = GeneratedDataset(self.model, self.diffusion, self.dataloader, self.mm_num_repeats, self.mm_num_samples, self.num_samples_limit, 
                                   self.dis_representation, self.diff_cfg, self.replication_times, self.visualizer, self.model_path, self.viz_dict, self.eval_type)   
        motion_loader = DataLoader(dataset, batch_size=1, drop_last=False, shuffle=False, num_workers=0)
        print(f"[DiffusionEval] Generated motion loader: {len(motion_loader)} completed.")
        return motion_loader

def evaluation(eval_data, eval_wrapper, diversity_times, mm_num_times, run_mm):
    
    motion_loaders = {}
    for motion_loader_name, motion_loader_getter in eval_data.items():
        motion_loader = motion_loader_getter()
        motion_loaders[motion_loader_name] = motion_loader

    # TODO: qualitative evaluations
    return None

if __name__ == '__main__':
    
    raise NotImplementedError("Not implemented yet.")

    # model, diffusion = create_model_and_diffusion(args, gen_loader)
    # state_dict = torch.load(args.model_path, map_location='cpu')
    # load_model_wo_clip(model, state_dict)
    # model.eval()
    # eval_data = {
    #     "test": lambda: DiffusionEval()
    # }
    # eval_wrapper = EvaluatorWrapper()
    # eval_dict = evaluation()
    # print(eval_dict)
    
