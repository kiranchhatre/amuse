
# AMUSE: Emotional Speech-driven 3D Body Animation via Disentangled Latent Diffusion
# CVPR 2024, Seattle, USA
# Kiran Chhatre, Radek Daněček, Nikos Athanasiou, Giorgio Becherini, Christopher Peters, Michael J. Black, Timo Bolkart

import os
import sys
import time
import json
import yaml
import torch
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from smac import Scenario, MultiFidelityFacade
from smac.intensifier.hyperband import Hyperband
from smac.intensifier.successive_halving import SuccessiveHalving

def main(args):
    
    tic = time.time()
    
    with open(args.cfg.name, "r+") as f:
        config = json.load(f)
        f.seek(0)
        json.dump(config, f, indent=4)
        f.truncate()   
    with open(args.wandb.name, "r+") as f:
        logger_cfg = json.load(f)
        
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda": torch.cuda.empty_cache()
    parallelism = config["TRAIN_PARAM"]["parallelism"]
    if parallelism == "dp": 
        assert device.type == "cuda"
        device = torch.device("cuda:0") # adjust if necessary

    # TODO: remove
    backup = config["TRAIN_PARAM"]["backup_experiment"] # Only implemented for latent_diffusion
    if backup:
        with open(str(Path(dirname, "configs", "backup_data.json")), "r+") as f: backup_cfg = json.load(f)
    else: backup_cfg = None

    # Raw data
    data_path = Path(dirname, "data/beat-rawdata-eng/beat_rawdata_english/")
    processed = Path(dirname, "data/BEAT-processed/")
    model_path = Path(dirname, "saved-models/")
    blender_resrc_path = Path(dirname, "models/diffusion/viz/")
    annotations_eng = Path(dirname, "data/beat_annotations_english/beat_cut_sem/")

    from scripts.trainer import trainer
    from models import allmodels
    from models.latent_diffusion.infer_ldm import PretrainedLPDM_v1
    from dm.dm import dm
    from dm.dataload import dataload, latdiff_long_collate_fn_v1
    from utils.misc import millify, move_lmdbs, fixseed
    from sweep_full_train import optimize_prior # SMAC support

    tag = config["TRAIN_PARAM"]["tag"] 
    if tag not in ["wav_mfcc", "wav_dtw_mfcc", "diffusion", "motionprior", "motionprior_long", "latent_diffusion", "audio_ae", "LPDM", "LPDM_infer"]:
        raise ValueError("Invalid tag name")
    display_model = config["TRAIN_PARAM"]["display_model_info"]
    display_model = True if EXEC_ON_CLUSTER else display_model
    pretrained_infer = config["TRAIN_PARAM"]["pretrained_infer"]
    debug = config["TRAIN_PARAM"]["debug"]
    debug = False if EXEC_ON_CLUSTER else debug

    # Caches
    new_cache = Path(cache_cluster_dirname, "data/BEAT-cache/") if EXEC_ON_CLUSTER else Path(cache_dirname, "BEAT-cache/")
    old_cache_id = config["TRAIN_PARAM"]["diffusion"]["lmdb_cache"]                 # for diffusion, prior*, latent_diffusion
    old_cache = Path(cache_cluster_dirname, "data", old_cache_id) if EXEC_ON_CLUSTER else Path(cache_dirname, old_cache_id)
    caches = [new_cache, old_cache]
    if tag in ["diffusion", "motionprior", "motionprior_long", "latent_diffusion"] and not backup:
        move_lmdbs(old_cache, EXEC_ON_CLUSTER, verbose=True)

    print(f"Experiment init: AMUSE, process: {os.getpid()}, running on: {os.uname()[1]}, distributed: {parallelism}, device: {device}, time: {time.asctime()}")
    fixseed(config["TRAIN_PARAM"]["seed"])
    if tag == "latent_diffusion": full_data = dm(data_path, annotations_eng, processed, latent_caches=caches, config=config, backup_cfg=backup_cfg)
    else: full_data = dm(data_path, annotations_eng, processed, config=config)

    # Preprocess
    if tag in ["motionprior", "motionprior_long", "latent_diffusion"] and config["TRAIN_PARAM"]["motionprior"]["emotional"]:
        full_data.emotionalpreprocess_v1(verbose=True) 
    else: full_data.preprocess(verbose=True)

    if args.fn[0] == "bvh2smplx_":
        assert config["DATA_PARAM"]["Bvh"]["bvh2smplbvh"], "smplx_extract must be True!"
        extract = config["DATA_PARAM"]["Bvh"]["smplx_extract_path"]
        full_data.beat2smplnpz(extract, blender_resrc_path) # Experimental, no support!
        import sys; sys.exit("AMUSE: Smplx npz conversion done! Import NPZ in Blender with SMPLX addon and specify correct fps.")

    sweep = any([config["TRAIN_PARAM"]["motionprior"]["sweep"],config["TRAIN_PARAM"]["latent_diffusion"]["sweep"]]) 
    # sweep support removed! (sweep over motionprior/ latent_diffusion)
    debug = True if sweep else debug
    if tag not in ["motionprior_long", "latent_diffusion"]: 
        bs = config["TRAIN_PARAM"][tag]["batch_size"] if not pretrained_infer else config["TRAIN_PARAM"][tag]["infer_batch_size"]
    else: bs = config["TRAIN_PARAM"][tag]["batch_size"]
    
    if args.fn[0] == "train_audio":  # Train speech disentanglment model     
                                                                                                                       
        assert config["TRAIN_PARAM"]["motion_extractor"]["use"] == False, "Motion extractor should be False!"
        DTW_align_dm = full_data.DTW_align_dm_ast()
        train_set = dataload(DTW_align_dm, config, tag, loader_type="train", shuffle_type=config["TRAIN_PARAM"][tag]["shuffle_type"]) # shuffle_type: random, takes, actors, all
        train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, drop_last=True)
        val_set = dataload(DTW_align_dm, config, tag, loader_type="val", shuffle_type=config["TRAIN_PARAM"][tag]["shuffle_type"])
        val_loader = DataLoader(val_set, batch_size=bs, shuffle=True, drop_last=True)
        model = allmodels[tag]
        if display_model:
            print(f"[DTW] model init")
            dtw_params = millify(model)
        model = model.to(device)
        trainer = trainer(config, device, train_loader, val_loader, model_path, tag, logger_cfg, model, debug=debug)
        trainer.train_dtw_ast()                                                                                                                        

    elif args.fn[0] in ["train_gesture", "infer_gesture", "prepare_data", "edit_gesture"]:
        
        if args.fn[0] == "prepare_data":  # Prepare LMDB dataloader
            if "ablation" in config["TRAIN_PARAM"]["wav_dtw_mfcc"]: audio_ablation = config['TRAIN_PARAM']['wav_dtw_mfcc']['ablation']
            else: audio_ablation = None
            latent_diffusion_dm = full_data.latent_diffusion_dm_v2(device, verbose=True, audio_ablation=audio_ablation)
            import sys; sys.exit("AMUSE: LMDB data prepared!")
    
        else: # Train gesture generation model
            
            assert (args.fn[0] == "train_gesture" and not pretrained_infer) or (args.fn[0] in ["infer_gesture", "edit_gesture"] and pretrained_infer), f"Arg: {args.fn[0]} and pretrained_infer: {pretrained_infer} mismatch!"
            smplx_data_training = config["TRAIN_PARAM"]["latent_diffusion"]["smplx_data"]
            if not pretrained_infer:
                assert smplx_data_training, "smplx_data must be True!"
                if "ablation" in config["TRAIN_PARAM"]["wav_dtw_mfcc"]: audio_ablation = config['TRAIN_PARAM']['wav_dtw_mfcc']['ablation']
                else: audio_ablation = None
                latent_diffusion_dm = full_data.latent_diffusion_dm_v2(device, verbose=True, audio_ablation=audio_ablation) 
                train_set = dataload(latent_diffusion_dm, config, tag, loader_type="train", shuffle_type=config["TRAIN_PARAM"]["motionprior"]["shuffle_type"])
                train_loader = DataLoader(train_set, batch_size=bs, shuffle=False, drop_last=True, 
                                        collate_fn=lambda x: latdiff_long_collate_fn_v1(x)) 
            else: 
                eval_loader = full_data.latent_diffusion_eval_dm_v1(smplx_data_training)  

            assert config["TRAIN_PARAM"]["motion_extractor"]["use"] == False, "Motion extractor should be False!"
            prior_model = allmodels["motionprior"]
            prior_model.setup(processed, config, prior_cfg=None, load_pretrained=False)
            prior_model.to(device)
            ldm_model = allmodels["latent_diffusion"]
            model_path_new = Path(dirname, "saved-models-new/") if EXEC_ON_CLUSTER else model_path
            if not pretrained_infer:
                combined_tag = "LPDM" # latent prior diffusion model
                ldm_model.setup(processed, config, device, tag, None, None, None, None, backup_cfg, combined_train=True)
                ldm_model.to(device)
                if display_model: print(f"[LAT-PRIOR-DIFF] model init"); latdiff_params = millify(prior_model, ldm_model)
                trainer = trainer(config, device, train_loader=train_loader, model_path=model_path_new, tag=combined_tag, 
                                logger_cfg=logger_cfg, model={"prior": prior_model, "ldm": ldm_model}, processed=processed, 
                                b_path=blender_resrc_path, EXEC_ON_CLUSTER=EXEC_ON_CLUSTER, debug=debug, pretrained_infer=pretrained_infer) 
                lmdb_cache = config["TRAIN_PARAM"]["diffusion"]["lmdb_cache"]
                ablat = config["TRAIN_PARAM"]["wav_dtw_mfcc"]["ablation_version"]                  # Dataset version: v0 gendered, v1 neutral
                trainer.train_prior_latdiff_forward_backward_v2(lmdb_id=lmdb_cache, audio_ablation=ablat,
                                                                verbose=True)                     
            else:
                
                
                combined_tag = "LPDM_infer"
                
                # Infer Baselines
                baseline = False
                modelversion = config["TRAIN_PARAM"]["wav_dtw_mfcc"]["ablation"]
                
                # Diff only
                diffonly = config["TRAIN_PARAM"]["test"]["diff_only"]
                
                # Audio list + Quick viz on short audio list / OR for latent plot
                audio_list = config["TRAIN_PARAM"]["test"]["audio_list"]["use"]
                
                # infer whole audio list, removed functionality
                short_audio_list = config["TRAIN_PARAM"]["test"]["audio_list"]["short_audio_list"]
                
                epoch_override = True
                audio_model_metrics = True
                
                if epoch_override:
                    ie = ["5000", "5200", "5400", "5600", "5800", "6000"] 
                    idx = 5  
                
                task = config["TRAIN_PARAM"]["baselines"]["renders"]["task"]
                if task == "Emotion_Control":
                    emotional_actors = [
                        "wayne",
                        "scott",
                        "solomon",
                        "lawrence",
                        "stewart",
                        "carla",
                        "sophie",
                        "catherine",
                        "miranda",
                        "kieks",
                        "zhao",
                        "lu",
                        "jorge",
                        "daiki", 
                        "ayana", 
                        "reamey",
                        "yingqing",
                        "katya" 
                    ]
                    chosen = f"[{emotional_actors[0]}]"
                    model_path_new = Path(config["TRAIN_PARAM"]["baselines"]["renders"][task]) / chosen[1:-1]
                    config["TRAIN_PARAM"]["test"]["emotion_control"]["actor"] = chosen
                    raise
                
                if epoch_override: eee = ie[idx] 
                else: 
                    assert config["TRAIN_PARAM"]["latent_diffusion"]["pretrained_prior_lpdm_e"] == config["TRAIN_PARAM"]["latent_diffusion"]["pretrained_ldm_lpdm_e"], "LPDM-combo epochs must be same!"
                    eee = config["TRAIN_PARAM"]["latent_diffusion"]["pretrained_prior_lpdm_e"]
                print(f"Inferring for epoch {eee}")
                config["TRAIN_PARAM"]["latent_diffusion"]["pretrained_prior_lpdm_e"] = eee
                config["TRAIN_PARAM"]["latent_diffusion"]["pretrained_ldm_lpdm_e"] = eee
                
                pretrained_lpdm = PretrainedLPDM_v1(prior_model)
                ldm_epoch = pretrained_lpdm.setup(config, device, processed, backup_cfg, EXEC_ON_CLUSTER, baseline, verbose=False, diffonly=diffonly) 
                trainer = trainer(config, device, train_loader=eval_loader, model_path=model_path_new, tag=combined_tag, 
                                logger_cfg=logger_cfg, model=pretrained_lpdm, processed=processed, metricsmodel=None,
                                b_path=blender_resrc_path, EXEC_ON_CLUSTER=EXEC_ON_CLUSTER, debug=debug, pretrained_infer=pretrained_infer) 
                trainer.eval_prior_latdiff_forward_backward_v1(baseline, ldm_epoch, audio_list, short_audio_list, modelversion=modelversion, ammetric=audio_model_metrics)

    print(f"AMUSE: ({args.fn[0]}) completed in: {(time.time()-tic)/3600} hrs")
    
if __name__ == "__main__":

    # Platforms   
    dirname = Path.cwd().parent
    cache_dirname = dirname
    cluster_dirname = Path.cwd().parents[2]
    cache_cluster_dirname = cluster_dirname  
    EXEC_ON_CLUSTER = False if dirname.name == "amuse" else True
    dirname = cluster_dirname if EXEC_ON_CLUSTER else dirname
    sys.path += [str(dirname)]

    parser = argparse.ArgumentParser(description='AMUSE')
    parser.add_argument("--cfg", default=str(Path(dirname, "configs/base_new.json")), 
                        type=argparse.FileType("r"), required=False, help="config file")
    parser.add_argument("--wandb", default=str(Path(dirname, "configs/logger.json")), 
                        type=argparse.FileType("r"), required=False, help="wandb")
    parser.add_argument("--fn", nargs="*", help="train_audio, train_gesture, infer_gesture, prepare_data, bvh2smplx_", required=True)
    args = parser.parse_args()
    
    # Overrides TODO Omegaconf
    override_yml = [f"{args.fn[0]}.yaml", "prior_o.yaml", "diff_o.yaml"]
    jsons_to_override = ["base_new.json", "prior_emotional_fing.json", "diff_latent_v2.json"]
    def merge_dicts(config, override_dict):
        def merge_recursive(original, override):
            for key, value in override.items():
                if isinstance(value, dict) and key in original and isinstance(original[key], dict):
                    merge_recursive(original[key], value)
                else:
                    original[key] = value
        if override_dict is None: return config
        else:
            result = config.copy()  
            merge_recursive(result, override_dict)
            return result
        
    for override, config in zip(override_yml, jsons_to_override):
        override_dict = yaml.safe_load(open(str(Path(dirname, "scripts/overrides", override))))
        config_dict = json.load(open(str(Path(dirname, "configs", config))))
        config_dict = merge_dicts(config_dict, override_dict)
        with open(str(Path(dirname, "configs", config)), "w") as f:
            json.dump(config_dict, f, indent=4)
    
    main(args)