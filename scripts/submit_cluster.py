


import os, subprocess, stat, datetime, argparse, json, pathlib

condor_template = """
executable = <<SCRIPTNAME>>
arguments = <<ARGS>>
error = <<PATH>>/<<JOBNAME>><<PROCESS_ID>>.err
output = <<PATH>>/<<JOBNAME>><<PROCESS_ID>>.out
log = <<PATH>>/<<JOBNAME>><<PROCESS_ID>>.log
request_memory = <<MEMORYMBS>>
request_cpus = <<CPU_COUNT>>
request_gpus = <<GPU_COUNT>>
<<REQUIREMENTS>>
<<CONCURRENCY>>
+MaxRunningPrice = <<MAX_PRICE>>
+RunningPriceExceededAction = "kill"
queue <<NJOBS>>
"""

script_template = """
source /home/<<USERNAME>>/.bashrc
source /home/<<USERNAME>>/anaconda3/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
conda activate <<ENV>>
export PYTHONPATH=$PYTHONPATH:<<REPO_ROOT>>
<<MODULES>>
<<PYTHON_BIN>> <<SCRIPT_NAME>> $@
# OUTFOLDER=$(cat out_folder.txt)
# ln -s $PWD $OUTFOLDER/submission 
# ln -s $OUTFOLDER results
# source deactivate
"""

parser = argparse.ArgumentParser(description='Execute on Cluster')
parser.add_argument("--cfg", default="cluster_settings.json", type=argparse.FileType("r"), required=False, help="config file")

def exec_on_cluster(config):  
    
    os.environ["CUDA_HOME"] = "/is/software/nvidia/cuda-" + config["Dir"]["cuda_version"]
    time = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
    submission_folder_name = time + "_" + str(hash(time)) + "_" + "submission"
    # submission_dir_cluster_side = os.path.join(config["Dir"]["submission_dir_cluster_side"], config["Dir"]["project"], config["Dir"]["model_type"], submission_folder_name) 
    submission_dir_cluster_side = os.path.join(config["Dir"]["submission_dir_cluster_side"], config["Dir"]["model_type"], submission_folder_name) 
    pathlib.Path(submission_dir_cluster_side).mkdir(parents=True, exist_ok=True) 
    logdir = config["Dir"]["logdir"]

    st = script_template 
    st = st.replace('<<REPO_ROOT>>', config["Dir"]["cluster_repo_dir"])
    st = st.replace('<<PYTHON_BIN>>', config["Constants"]["python_bin"])
    run_script = os.path.join(config["Dir"]["cluster_repo_dir"], config["Dir"]["cluster_script_path"])
    st = st.replace('<<SCRIPT_NAME>>', run_script)
    st = st.replace('<<ENV>>', config["Dir"]["env"])
    st = st.replace('<<USERNAME>>', config["Constants"]["username"])
    # st = st.replace('<<MODULES>>', "module load cuda/" + config["Dir"]["cuda_version"])
    st = st.replace('<<MODULES>>', "")
    # st = st.replace('$@', project_cfg["shell_args"])
    script_fname = os.path.join(submission_dir_cluster_side, 'run.sh') 
    
    cs = condor_template
    cs = cs.replace('<<PATH>>', logdir)
    # script_args = "" if len(project_cfg["shell_args"])!=0 else project_cfg["one_liner_args"]
    # cs = cs.replace('<<ARGS>>', script_args)
    # for empty args
    cs = cs.replace('<<ARGS>>', "")
    cs = cs.replace('<<SCRIPTNAME>>', os.path.basename(script_fname))
    cs = cs.replace('<<JOBNAME>>', config["Dir"]["job_name"])
    cs = cs.replace('<<CPU_COUNT>>', str(int(config["Compute"]["cpus"])))
    cs = cs.replace('<<GPU_COUNT>>', str(int(config["Compute"]["gpus"])))
    cs = cs.replace('<<MEMORYMBS>>', str(int(config["Compute"]["mem_gb"] * 1024)))
    cs = cs.replace('<<MAX_TIME>>', str(int(config["Compute"]["max_time_h"] * 3600))) 
    cs = cs.replace('<<MAX_PRICE>>', str(int(config["Compute"]["max_price"])))
    cs = cs.replace('<<NJOBS>>', str(config["Compute"]["num_jobs"]))


    if config["Compute"]["num_jobs"]>1:
        cs = cs.replace('<<PROCESS_ID>>', ".$(Process)")
    else:
        cs = cs.replace('<<PROCESS_ID>>', "")

    requirements = []

    gpu_mem_requirement_mb = config["Compute"]["gpu_memory_min_gb"] * 1024
    requirements += [f"(TARGET.CUDAGlobalMemoryMb>={gpu_mem_requirement_mb})"]

    # gpu_mem_requirement_mb_max = config["Compute"]["gpu_memory_max_gb"] * 1024
    # requirements += [f"( TARGET.CUDAGlobalMemoryMb<={gpu_mem_requirement_mb_max} )"]

    cuda_model = config["Compute"]["device_model"]
    if cuda_model:
        requirements += [f"""(TARGET.CUDADeviceName=="{cuda_model}")"""]
    
    node = config["Compute"]["node"]
    if node:
        requirements += [f"(UtsnameNodename=={node})"]

    if len(requirements) > 0:
        requirements = " && ".join(requirements)
        requirements = "requirements=" + requirements

    cs = cs.replace('<<REQUIREMENTS>>', requirements)
    condor_fname = os.path.join(submission_dir_cluster_side, 'run.condor')
    
    concurrency_string = ""
    concurrency_tag = config["Compute"]["concurrency_tag"]
    max_concurrent_jobs = config["Compute"]["max_concurrent_jobs"]
    if concurrency_tag != 0 and max_concurrent_jobs != 0:
        concurrency_limits = 10000 // max_concurrent_jobs
        concurrency_string += f"concurrency_limits = user.{concurrency_tag}:{concurrency_limits}"
    cs = cs.replace('<<CONCURRENCY>>', concurrency_string)
    
    # TODO: add wandb support

    # write files
    with open(script_fname, 'w') as fp:
        fp.write(st)
    os.chmod(script_fname, stat.S_IXOTH | stat.S_IWOTH | stat.S_IREAD | stat.S_IEXEC | stat.S_IXUSR | stat.S_IRUSR)  # make executable
    with open(condor_fname, 'w') as fp:
        fp.write(cs)
    os.chmod(condor_fname, stat.S_IXOTH | stat.S_IWOTH | stat.S_IREAD | stat.S_IEXEC | stat.S_IXUSR | stat.S_IRUSR)  # make executable

    bid = config["Compute"]["bid"]
    cmd = f'cd {submission_dir_cluster_side} && ' \
            f'mkdir {logdir} && ' \
            f'chmod +x {os.path.basename(script_fname)} && ' \
            f'chmod +x {os.path.basename(condor_fname)} && ' \
            f'condor_submit_bid {bid} {os.path.basename(condor_fname)}'

    print("Called the following on the cluster: ")
    print(cmd)
    subprocess.call(["ssh", "%s@login.cluster.is.localnet" % (config["Constants"]["username"],)] + [cmd])
    print("Done")

if __name__ == "__main__":
    
    args = parser.parse_args()
    with open(args.cfg.name, "r+") as f:
        config = json.load(f)
        # config["Dir"]["ROOT"] = os.getcwd()
        # config["Dir"]["CONFIG_ROOT"] = args.cfg.name
        # f.seek(0)
        # json.dump(config, f, indent=4)
        # f.truncate()
 
    exec_on_cluster(config) 