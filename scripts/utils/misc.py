# Adapted from https://stackoverflow.com/a/62508086
# https://stackoverflow.com/a/3155023

import os
import math
import shutil
import torch
import random
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    return total_params

millnames = ['',' Thousand',' Million',' Billion',' Trillion']

def millify(model, model1=None):
    if model1 is not None: print("======= Model 1/2 =======")
    else: print("======= Model 1/1 =======")
    total_params = count_parameters(model)
    n = float(total_params)
    millidx = max(0,min(len(millnames)-1,
                        int(math.floor(0 if n == 0 else math.log10(abs(n))/3))))
    print(f"Trainable {n / 10**(3 * millidx)} {millnames[millidx]} parameters")
    if model1 is not None:
        print("======= Model 2/2 =======")
        total_params1 = count_parameters(model1)
        n1 = float(total_params1)
        millidx1 = max(0,min(len(millnames)-1,
                            int(math.floor(0 if n1 == 0 else math.log10(abs(n1))/3))))
        # combined params
        n_combined = n + n1
        millidx_combined = max(0,min(len(millnames)-1,
                                     int(math.floor(0 if n_combined == 0 else math.log10(abs(n_combined))/3))))
        print(f"Trainable {n1 / 10**(3 * millidx1)} {millnames[millidx1]} parameters")
        print(f"Combined Trainable {n_combined / 10**(3 * millidx_combined)} {millnames[millidx_combined]} parameters")
    return 

def move_lmdbs(old_cache, EXEC_ON_CLUSTER, verbose=False):
    cache_name = old_cache.name
    exec_on_cluster_from_path = True if old_cache.parents[1].name == "data" else False
    if not cache_name == "NEW_CACHE":
        assert exec_on_cluster_from_path == EXEC_ON_CLUSTER, f"[MOVE LMDBs/ MISMATCH] exec_on_cluster_from_path: {exec_on_cluster_from_path}, EXEC_ON_CLUSTER: {EXEC_ON_CLUSTER}"
    if verbose:
        print("[MOVE LMDBs] old cache: ", cache_name , " on cluster: " if EXEC_ON_CLUSTER else " on local machine")
    if cache_name == "NEW_CACHE":
        if verbose: print(f"[MOVE LMDBs] new cache creation, skipping...")
        pass
    else:
        CACHE_DIR = old_cache.parents[0] 
        all_caches = list(CACHE_DIR.iterdir())
        relevant_caches = [c for c in all_caches if c.name.startswith(cache_name)]
        # print(f"relevant caches: {relevant_caches}, all caches: {all_caches}")
        if EXEC_ON_CLUSTER:
            if len(relevant_caches) < 1: raise Exception(f"Found {len(relevant_caches)} relevant caches for {cache_name}, expected 1 or 3 or 4. Run locally to generate the split caches.")
            else: 
                if verbose: print(f"[MOVE LMDBs] Found relevant caches for {cache_name}, skipping...")
        else:
            if len(relevant_caches) == 0:
                cluster_4cp_cache_dir = Path("/is/cluster/kchhatre/Work/Datasets/BEAT/BEAT-cache")          # remove hardcoding
                cluster_4cp_all_caches = list(cluster_4cp_cache_dir.iterdir())
                cluster_4cp_relevant_caches = [c for c in cluster_4cp_all_caches if c.name.startswith(cache_name)]
                assert len(cluster_4cp_relevant_caches) in [1,2,3,4], f"Found {len(cluster_4cp_relevant_caches)} relevant caches for {cache_name}, expected 1, 2, 3 or 4. Debug!"
                if len(all_caches) != 0:
                    if verbose: print(f"[MOVE LMDBs] found irrelevant caches, deleting...")
                    for cache_dir in tqdm(all_caches, desc=f"[MOVE LMDBs] deleting irrelevant caches...", leave=False):
                        shutil.rmtree(cache_dir)
                if verbose: print(f"[MOVE LMDBs] found relevant caches, copying...")
                for cache_dir in tqdm(cluster_4cp_relevant_caches, desc=f"[MOVE LMDBs] moving {cache_name}...", leave=False):
                    shutil.copytree(cache_dir, CACHE_DIR/cache_dir.name, dirs_exist_ok=True)
            elif len(relevant_caches) in [1,2,3,4]:
                excess_caches = list(set(all_caches)^set(relevant_caches))
                if len(excess_caches) != 0:
                    if verbose: print(f"[MOVE LMDBs] found irrelevant and relevant caches, deleting irrelevant...")
                    for cache_dir in tqdm(excess_caches, desc=f"[MOVE LMDBs] deleting {cache_name}...", leave=False):
                        shutil.rmtree(cache_dir)
                else:
                    if verbose: print(f"[MOVE LMDBs] found relevant caches only, skipping...")
            else: raise Exception(f"Found {len(relevant_caches)} relevant caches for {cache_name}, expected 0, 1, 2, 3 or 4. Debug!")
                    
def fixseed(seed):
    os.environ["PYTHONWARNINGS"] = "ignore"
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)