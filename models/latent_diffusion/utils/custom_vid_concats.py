


# NOTE HACK: this script does not work on cluster, only locally.
# conversion: mp4 --> ts
# command: ffmpeg -i input.mp4 -c copy -bsf:v h264_mp4toannexb -f mpegts out.ts
# error on cluster: [h264_mp4toannexb @ 0x564461526680] Codec 'mpeg4' (12) is not supported by the bitstream filter 'h264_mp4toannexb'. Supported codecs are: h264 (27) 
# Error initializing bitstream filter: h264_mp4toannexb


"""
main: /is/cluster/kchhatre/Work/code/disentangled-s2g/saved-models-new/amusepp_all_tests

dir further: viz/***/rep0
*** = style_Xemo_transfer_***
*** = emotion_control_***

LPDM_20231023-200335_actors_smplx
LPDM_20231023-195616_actors_smplx
LPDM_20231023-195919_actors_smplx
LPDM_20231023-195413_actors_smplx

EC:
LPDM_20231023-195413_actors_smplx
LPDM_20231023-195919_actors_smplx
LPDM_20231025-165555_actors_smplx
LPDM_20231025-165844_actors_smplx
"""

import time
import subprocess
from pathlib import Path

def concat_style_Xemo_transfer(subdir):
    # rootpath = Path("/is/cluster/kchhatre/Work/code/disentangled-s2g/saved-models-new/amusepp_all_tests") # cluster
    rootpath = Path("/home/kchhatre/Work/code/disentangled-s2g/saved-models/amusepp_all_tests") # local
    rootpath = rootpath / subdir / "viz"
    rootpath = rootpath.glob("*")
    rootpath = [x for x in rootpath if x.is_dir() and "style_Xemo_transfer" in x.name][0]
    video_dump_r = rootpath / "rep0"
    assert video_dump_r.exists(), "Invalid path: [%s]" % video_dump_r
    
    # video_dump_r = self.model_path / "viz" / f"style_Xemo_transfer_{run_info}_{self.stamp}_E{ldm_epoch}" / f"rep{rep_i}"
    # assert self.viz_type in ["CaMN"], "[LDM EVAL] Invalid viz type: [%s]" % self.viz_type
    # for i, sample_dict in enumerate(rst):
    #     print(f"VISUALIZATION: STYLE TRANSFER {i} =====>")
    #     video_dump = video_dump_r / f"rst_{i}"
    #     self.visualizer.animate_ldm_sample_v2(sample_dict, video_dump)
    
    print(f"VISUALIZATION: STYLE TRANSFER COMBINED =====>")
    
    rst_dirs = [f for f in video_dump_r.iterdir() if f.is_dir()]
    rst_dirs = sorted(rst_dirs, key=lambda x: int(x.name.split("_")[-1]))
    take_1_gen, take_2_gen = rst_dirs[0:4], rst_dirs[4:8]
    min_t1_mp4 = min([len(list(gen.glob("*/*.mp4"))) for gen in take_1_gen]) 
    min_t2_mp4 = min([len(list(gen.glob("*/*.mp4"))) for gen in take_2_gen]) 
    
    combined_video_path = video_dump_r / "combined"
    combined_video_path.mkdir(parents=True, exist_ok=True)
    
    for i in range(min_t1_mp4):
        # TODO: do not generate seq 0 and -1, during pauses and no driving speech
        combined_video_file = combined_video_path / f"combined_0_set_{i}.mp4"
        
        vid1 = next(Path(take_1_gen[0], f"seq_{i}").glob("*.mp4"))         
        vid2 = next(Path(take_1_gen[1], f"seq_{i}").glob("*.mp4"))
        vid3 = next(Path(take_1_gen[2], f"seq_{i}").glob("*.mp4"))
        vid4 = next(Path(take_1_gen[3], f"seq_{i}").glob("*.mp4"))
        
        # make 2x2 grid of above 4 videos with audio from 1st video
        _ = subprocess.call([
            "ffmpeg", "-i", str(vid1), "-i", str(vid2), "-i", str(vid3), "-i", str(vid4), "-filter_complex", "[0:v][1:v]hstack[top];[2:v][3:v]hstack[bottom];[top][bottom]vstack,format=yuv420p[v]", "-map", "[v]", str(combined_video_file)
        ])
        
        # add audio from vid1 to combined video
        combined_video_file_Audio = combined_video_path / f"combined_0_set_{i}_audio_1.mp4"
        _ = subprocess.call([
            "ffmpeg", "-i", str(combined_video_file), "-i", str(vid1), "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0", "-shortest", str(combined_video_file_Audio)
        ])
        
        # add audio from vid2 to combined video
        combined_video_file_Audio = combined_video_path / f"combined_0_set_{i}_audio_2.mp4"
        _ = subprocess.call([
            "ffmpeg", "-i", str(combined_video_file), "-i", str(vid2), "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0", "-shortest", str(combined_video_file_Audio)
        ])
        
        # delete combined_video_file
        combined_video_file.unlink()
        
        # # Old code
        # for v in [vid1, vid2, vid3, vid4]:
        #     _ = subprocess.call([
        #         "ffmpeg", "-i", str(v), "-c", "copy", "-bsf:v", "h264_mp4toannexb", "-f", "mpegts", str(v.with_suffix(".ts"))
        #     ])

        # _ = subprocess.call([
        #     "ffmpeg", "-i", f"concat:{str(vid1.with_suffix('.ts'))}|{str(vid2.with_suffix('.ts'))}|{str(vid3.with_suffix('.ts'))}|{str(vid4.with_suffix('.ts'))}", "-c", "copy", "-bsf:a", "aac_adtstoasc", str(combined_video_file)
        # ])
        
        # for v in [vid1, vid2, vid3, vid4]: v.with_suffix(".ts").unlink()
    
    for i in range(min_t2_mp4):
        
        combined_video_file = combined_video_path / f"combined_1_set_{i}.mp4"
        
        vid1 = next(Path(take_2_gen[0], f"seq_{i}").glob("*.mp4"))         
        vid2 = next(Path(take_2_gen[1], f"seq_{i}").glob("*.mp4"))
        vid3 = next(Path(take_2_gen[2], f"seq_{i}").glob("*.mp4"))
        vid4 = next(Path(take_2_gen[3], f"seq_{i}").glob("*.mp4"))
        
        _ = subprocess.call([
            "ffmpeg", "-i", str(vid1), "-i", str(vid2), "-i", str(vid3), "-i", str(vid4), "-filter_complex", "[0:v][1:v]hstack[top];[2:v][3:v]hstack[bottom];[top][bottom]vstack,format=yuv420p[v]", "-map", "[v]", str(combined_video_file)
        ])
        combined_video_file_Audio = combined_video_path / f"combined_1_set_{i}_audio_1.mp4"
        _ = subprocess.call([
            "ffmpeg", "-i", str(combined_video_file), "-i", str(vid1), "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0", "-shortest", str(combined_video_file_Audio)
        ])
        combined_video_file_Audio = combined_video_path / f"combined_1_set_{i}_audio_2.mp4"
        _ = subprocess.call([
            "ffmpeg", "-i", str(combined_video_file), "-i", str(vid2), "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0", "-shortest", str(combined_video_file_Audio)
        ])
        combined_video_file.unlink()
        
        # # Old Code
        # for v in [vid1, vid2, vid3, vid4]:
        #     _ = subprocess.call([
        #         "ffmpeg", "-i", str(v), "-c", "copy", "-bsf:v", "h264_mp4toannexb", "-f", "mpegts", str(v.with_suffix(".ts"))
        #     ])

        # _ = subprocess.call([
        #     "ffmpeg", "-i", f"concat:{str(vid1.with_suffix('.ts'))}|{str(vid2.with_suffix('.ts'))}|{str(vid3.with_suffix('.ts'))}|{str(vid4.with_suffix('.ts'))}", "-c", "copy", "-bsf:a", "aac_adtstoasc", str(combined_video_file)
        # ])
        
        # for v in [vid1, vid2, vid3, vid4]: v.with_suffix(".ts").unlink()
    rep_i = 0
    print(f"END VISUALIZATION: STYLE X-EMO TRANSFER {rep_i+1} =====>")

# lists = ["LPDM_20231023-200335_actors_smplx"]
# for l in lists:
#     concat_style_Xemo_transfer(l)
    
def concat_emotion_control(subdir):
    
    rootpath = Path("/is/cluster/kchhatre/Work/code/disentangled-s2g/saved-models-new/amusepp_all_tests")
    rootpath = rootpath / subdir / "viz"
    rootpath = rootpath.glob("*")
    rootpath = [x for x in rootpath if x.is_dir() and "emotion_control" in x.name][0]
    video_dump_r = rootpath / "rep0"
    assert video_dump_r.exists(), "Invalid path: [%s]" % video_dump_r
    
    print(f"VISUALIZATION: EMOTION CONTROL COMBINED =====>") 
                        
    rst_dirs = [f for f in video_dump_r.iterdir() if f.is_dir()]
    assert len(rst_dirs) == 64, "[LDM EVAL] Invalid number of rst dirs: [%d]" % len(rst_dirs) # 8 original emotions * 8 swapped emotions = 64
    rst_dirs = sorted(rst_dirs, key=lambda x: int(x.name.split("_")[-1]))
    
    take_0_gen = rst_dirs[0:8]
    take_1_gen = rst_dirs[8:16]
    take_2_gen = rst_dirs[16:24]
    take_3_gen = rst_dirs[24:32]
    take_4_gen = rst_dirs[32:40]
    take_5_gen = rst_dirs[40:48]
    take_6_gen = rst_dirs[48:56]
    take_7_gen = rst_dirs[56:64]
    
    min_t0_mp4 = min([len(list(gen.glob('*/*.mp4'))) for gen in take_0_gen])
    min_t1_mp4 = min([len(list(gen.glob('*/*.mp4'))) for gen in take_1_gen])
    min_t2_mp4 = min([len(list(gen.glob('*/*.mp4'))) for gen in take_2_gen])
    min_t3_mp4 = min([len(list(gen.glob('*/*.mp4'))) for gen in take_3_gen])
    min_t4_mp4 = min([len(list(gen.glob('*/*.mp4'))) for gen in take_4_gen])
    min_t5_mp4 = min([len(list(gen.glob('*/*.mp4'))) for gen in take_5_gen])
    min_t6_mp4 = min([len(list(gen.glob('*/*.mp4'))) for gen in take_6_gen])
    min_t7_mp4 = min([len(list(gen.glob('*/*.mp4'))) for gen in take_7_gen])

    combined_video_path = video_dump_r / "combined"
    combined_video_path.mkdir(parents=True, exist_ok=True)
    
    for ii in range(8):
        for jj in range(eval(f"min_t{ii}_mp4")):
            combined_video_file = combined_video_path / f"combined_{ii}_set_{jj}.mp4"
            
            # create grid of 8 videos
            vid1 = next(Path(eval(f"take_{ii}_gen[0]"), f"seq_{jj}").glob("*.mp4"))
            vid2 = next(Path(eval(f"take_{ii}_gen[1]"), f"seq_{jj}").glob("*.mp4"))
            vid3 = next(Path(eval(f"take_{ii}_gen[2]"), f"seq_{jj}").glob("*.mp4"))
            vid4 = next(Path(eval(f"take_{ii}_gen[3]"), f"seq_{jj}").glob("*.mp4"))
            vid5 = next(Path(eval(f"take_{ii}_gen[4]"), f"seq_{jj}").glob("*.mp4"))
            vid6 = next(Path(eval(f"take_{ii}_gen[5]"), f"seq_{jj}").glob("*.mp4"))
            vid7 = next(Path(eval(f"take_{ii}_gen[6]"), f"seq_{jj}").glob("*.mp4"))
            vid8 = next(Path(eval(f"take_{ii}_gen[7]"), f"seq_{jj}").glob("*.mp4"))
            _ = subprocess.call([
                "ffmpeg", "-i", str(vid1), "-i", str(vid2), "-i", str(vid3), "-i", str(vid4), "-i", str(vid5), "-i", str(vid6), "-i", str(vid7), "-i", str(vid8), "-filter_complex", "[0:v][1:v]hstack[top1];[2:v][3:v]hstack[top2];[4:v][5:v]hstack[top3];[6:v][7:v]hstack[top4];[top1][top2]vstack[bottom1];[top3][top4]vstack[bottom2];[bottom1][bottom2]vstack,format=yuv420p[v]", "-map", "[v]", str(combined_video_file)
            ])   
            combined_video_file_w_audio = combined_video_path / f"combined_{ii}_set_{jj}_audio_1.mp4"
            _ = subprocess.call([
                "ffmpeg", "-i", str(combined_video_file), "-i", str(vid1), "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0", "-shortest", str(combined_video_file_w_audio)
            ])
            combined_video_file.unlink()
            
            # # Old Code
            # for kk in range(8):
            #     vid = next(Path(eval(f"take_{ii}_gen[{kk}]"), f"seq_{jj}").glob("*.mp4"))
            #     _ = subprocess.call([
            #         "ffmpeg", "-y", "-i", str(vid), "-c", "copy", "-bsf:v", "h264_mp4toannexb", "-f", "mpegts", str(vid.with_suffix(".ts"))
            #     ])

            # vid1 = next(Path(eval(f"take_{ii}_gen[0]"), f"seq_{jj}").glob("*.ts"))
            # vid2 = next(Path(eval(f"take_{ii}_gen[1]"), f"seq_{jj}").glob("*.ts"))   
            # vid3 = next(Path(eval(f"take_{ii}_gen[2]"), f"seq_{jj}").glob("*.ts"))   
            # vid4 = next(Path(eval(f"take_{ii}_gen[3]"), f"seq_{jj}").glob("*.ts"))   
            # vid5 = next(Path(eval(f"take_{ii}_gen[4]"), f"seq_{jj}").glob("*.ts"))   
            # vid6 = next(Path(eval(f"take_{ii}_gen[5]"), f"seq_{jj}").glob("*.ts"))   
            # vid7 = next(Path(eval(f"take_{ii}_gen[6]"), f"seq_{jj}").glob("*.ts"))   
            # vid8 = next(Path(eval(f"take_{ii}_gen[7]"), f"seq_{jj}").glob("*.ts"))
            
            # _ = subprocess.call([
            #     "ffmpeg", "-i", f"concat:{str(vid1)}|{str(vid2)}|{str(vid3)}|{str(vid4)}|{str(vid5)}|{str(vid6)}|{str(vid7)}|{str(vid8)}", "-c", "copy", "-bsf:a", "aac_adtstoasc", str(combined_video_file) 
            # ])
            
            # for kk in range(8):
            #     ts_file = next(Path(eval(f"take_{ii}_gen[{kk}]"), f"seq_{jj}").glob("*.ts"))
            #     _ = subprocess.call(["rm", str(ts_file)])

    print(f"VISUALIZATION: EMOTION CONTROL ORIGINAL COMBINED =====>")
    orig_gens = [take_0_gen[0], take_1_gen[0], take_2_gen[0], take_3_gen[0], take_4_gen[0], take_5_gen[0], take_6_gen[0], take_7_gen[0]]
    min_orig_gens_mp4 = min([len(list(gen.glob('*/*.mp4'))) for gen in orig_gens])
    orig_gens_video_path = video_dump_r / "orig_combined"
    orig_gens_video_path.mkdir(parents=True, exist_ok=True)
    
    for kk in range(min_orig_gens_mp4):
        video_file = orig_gens_video_path / f"orig_combined_emotion_{kk}.mp4"
        
        vid1 = next(Path(orig_gens[0], f"seq_{kk}").glob("*.mp4"))
        vid2 = next(Path(orig_gens[1], f"seq_{kk}").glob("*.mp4"))
        vid3 = next(Path(orig_gens[2], f"seq_{kk}").glob("*.mp4"))
        vid4 = next(Path(orig_gens[3], f"seq_{kk}").glob("*.mp4"))
        vid5 = next(Path(orig_gens[4], f"seq_{kk}").glob("*.mp4"))
        vid6 = next(Path(orig_gens[5], f"seq_{kk}").glob("*.mp4"))
        vid7 = next(Path(orig_gens[6], f"seq_{kk}").glob("*.mp4"))
        vid8 = next(Path(orig_gens[7], f"seq_{kk}").glob("*.mp4"))
        _ = subprocess.call([
            "ffmpeg", "-i", str(vid1), "-i", str(vid2), "-i", str(vid3), "-i", str(vid4), "-i", str(vid5), "-i", str(vid6), "-i", str(vid7), "-i", str(vid8), "-filter_complex", "[0:v][1:v]hstack[top1];[2:v][3:v]hstack[top2];[4:v][5:v]hstack[top3];[6:v][7:v]hstack[top4];[top1][top2]vstack[bottom1];[top3][top4]vstack[bottom2];[bottom1][bottom2]vstack,format=yuv420p[v]", "-map", "[v]", str(video_file)
        ])
        video_file_w_audio = orig_gens_video_path / f"orig_combined_emotion_{kk}_audio_1.mp4"
        _ = subprocess.call([
            "ffmpeg", "-i", str(video_file), "-i", str(vid1), "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0", "-shortest", str(video_file_w_audio)
        ])
        video_file.unlink()
        
        # # Old Code
        # for ll in range(8):
        #     vid = next(Path(orig_gens[ll], f"seq_{kk}").glob("*.mp4"))
        #     _ = subprocess.call([
        #         "ffmpeg", "-y", "-i", str(vid), "-c", "copy", "-bsf:v", "h264_mp4toannexb", "-f", "mpegts", str(vid.with_suffix(".ts"))
        #     ])
        # vid1 = next(Path(orig_gens[0], f"seq_{kk}").glob("*.ts"))
        # vid2 = next(Path(orig_gens[1], f"seq_{kk}").glob("*.ts"))
        # vid3 = next(Path(orig_gens[2], f"seq_{kk}").glob("*.ts"))
        # vid4 = next(Path(orig_gens[3], f"seq_{kk}").glob("*.ts"))
        # vid5 = next(Path(orig_gens[4], f"seq_{kk}").glob("*.ts"))
        # vid6 = next(Path(orig_gens[5], f"seq_{kk}").glob("*.ts"))
        # vid7 = next(Path(orig_gens[6], f"seq_{kk}").glob("*.ts"))
        # vid8 = next(Path(orig_gens[7], f"seq_{kk}").glob("*.ts"))
        # _ = subprocess.call([
        #     "ffmpeg", "-i", f"concat:{str(vid1)}|{str(vid2)}|{str(vid3)}|{str(vid4)}|{str(vid5)}|{str(vid6)}|{str(vid7)}|{str(vid8)}", "-c", "copy", "-bsf:a", "aac_adtstoasc", str(video_file) 
        # ])
        # for zz in range(8):
        #     ts_file = next(Path(orig_gens[zz], f"seq_{kk}").glob("*.ts"))
        #     _ = subprocess.call(["rm", str(ts_file)])
    rep_i = 0
    print(f"END VISUALIZATION: EMOTION CONTROL {rep_i+1} =====>")
  


############### CUSTOM CONCAT TASK FOR EMOTION EDITING #########################

basefolder = Path("/is/cluster/kchhatre/Work/code/disentangled-s2g/saved-models-new/amusepp_all_tests/SUPMAT-LPDM_20231028-210758_actors_smplx/viz")



foldernames = []


all_foldersss= [
"emotion_control_ayana_first_20231121-002848_E6000,\
emotion_control_ayana_last_20231121-003000_E6000,\
emotion_control_jorge_first_20231121-002532_E6000,\
emotion_control_jorge_last_20231121-002709_E6000,\
emotion_control_kieks_first_20231121-001944_E6000,\
emotion_control_kieks_last_20231121-002402_E6000,\
emotion_control_lawrence_first_20231120-234152_E6000,\
emotion_control_lawrence_last_20231120-234625_E6000,\
emotion_control_lu_first_20231120-233046_E6000,\
emotion_control_lu_last_20231120-233829_E6000,\
emotion_control_solomon_first_20231120-222800_E6000,\
emotion_control_solomon_last_20231120-223007_E6000,\
emotion_control_stewart_first_20231120-223514_E6000,\
emotion_control_stewart_last_20231120-223654_E6000,\
emotion_control_wayne_first_20231120-222120_E6000,\
emotion_control_wayne_last_20231120-222225_E6000,\
emotion_control_zhao_first_20231120-223126_E6000,\
emotion_control_zhao_last_20231120-223253_E6000"]

foldernames = all_foldersss[0].split(",")


for foldername in foldernames:
    
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print(f"Processing: {foldername}")
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

    recons = basefolder / foldername / "rep0"
    which_take = foldername.split("_")[-3] # first, last, random
    subject = foldername.split("_")[-4] # lu, daiki, lawrence

    recons_dir_list = [f for f in recons.iterdir() if f.is_dir()]
    recons_dir_list = [f for f in recons_dir_list if "combined" not in f.name]
    recons_dir_list = sorted(recons_dir_list, key=lambda x: int(x.name.split("_")[-1]))

    GT = Path("/ps/scratch/kchhatre/Work/dis_s2g_backup/smplx_mosh_npz_blue_renders_half")
    GT_all_neutrals = Path("/is/cluster/work/kchhatre/Work/Dataset/BEAT/mosh_neutral_render")
    GT_vids = [f for f in GT.glob(f"{subject}_*.mp4")]
    GT_all_neutrals_vids = [f for f in GT_all_neutrals.glob(f"{subject}_*.mp4")]

    recons_dir_list = recons_dir_list[0:8] # neutral
    # recons_dir_list = recons_dir_list[32:40] # contempt

    dirs_in_each_recons_dir_list = [len(list(f.glob("*"))) for f in recons_dir_list]
    min_dirs = min(dirs_in_each_recons_dir_list)

    neutral_recons_vids = [f for f in recons_dir_list[0].glob("*/*.mp4")] 
    happy_recons_vids = [f for f in recons_dir_list[1].glob("*/*.mp4")]
    angry_recons_vids = [f for f in recons_dir_list[2].glob("*/*.mp4")]
    sad_recons_vids = [f for f in recons_dir_list[3].glob("*/*.mp4")]  
    contempt_recons_vids = [f for f in recons_dir_list[4].glob("*/*.mp4")]  
    surprise_recons_vids = [f for f in recons_dir_list[5].glob("*/*.mp4")]
    feared_recons_vids = [f for f in recons_dir_list[6].glob("*/*.mp4")] 
    disgusted_recons_vids = [f for f in recons_dir_list[7].glob("*/*.mp4")]    

    # contempt_recons_vids = [f for f in recons_dir_list[0].glob("*/*.mp4")] # contempt
    # neutral_recons_vids = [f for f in recons_dir_list[1].glob("*/*.mp4")] # neutral
    # happy_recons_vids = [f for f in recons_dir_list[2].glob("*/*.mp4")] # happy
    # angry_recons_vids = [f for f in recons_dir_list[3].glob("*/*.mp4")]   # angry
    # sad_recons_vids = [f for f in recons_dir_list[4].glob("*/*.mp4")]  # sad
    # surprise_recons_vids = [f for f in recons_dir_list[5].glob("*/*.mp4")] # surprise
    # feared_recons_vids = [f for f in recons_dir_list[6].glob("*/*.mp4")] # feared
    # disgusted_recons_vids = [f for f in recons_dir_list[7].glob("*/*.mp4")]    # disgusted

    neutral_recons_vids = sorted(neutral_recons_vids, key=lambda x: int(x.parent.name.split("_")[-1]))
    happy_recons_vids = sorted(happy_recons_vids, key=lambda x: int(x.parent.name.split("_")[-1]))
    angry_recons_vids = sorted(angry_recons_vids, key=lambda x: int(x.parent.name.split("_")[-1]))
    sad_recons_vids = sorted(sad_recons_vids, key=lambda x: int(x.parent.name.split("_")[-1]))
    contempt_recons_vids = sorted(contempt_recons_vids, key=lambda x: int(x.parent.name.split("_")[-1]))
    surprise_recons_vids = sorted(surprise_recons_vids, key=lambda x: int(x.parent.name.split("_")[-1]))
    feared_recons_vids = sorted(feared_recons_vids, key=lambda x: int(x.parent.name.split("_")[-1]))
    disgusted_recons_vids = sorted(disgusted_recons_vids, key=lambda x: int(x.parent.name.split("_")[-1]))

    train_first_takes = ["0_9_9", "0_65_65", "0_73_73", "0_81_81", "0_87_87", "0_95_95", "0_103_103", "0_111_111"]
    train_last_takes = ["0_10_10", "0_66_66", "0_74_74", "0_82_82", "0_88_88", "0_96_96", "0_104_104", "0_112_112"]
    train_random_takes = ["0_10_10", "0_65_65", "0_74_74", "0_81_81", "0_88_88", "0_95_95", "0_104_104", "0_111_111"]
    take_element2takes = {
        "first": train_first_takes,
        "last": train_last_takes,
        "random": train_random_takes
    }

    takes2use = take_element2takes[which_take]
    GT_vids_takes = [f for f in GT_vids if any([t in f.name for t in takes2use])]
    GT_all_neutrals_vids_takes = [f for f in GT_all_neutrals_vids if any([t in f.name for t in takes2use])]

    # GT_neutrals = [f for f in GT_vids_takes if takes2use[0] in f.name]
    GT_neutrals = [f for f in GT_all_neutrals_vids_takes if takes2use[0] in f.name]
    GT_happys = [f for f in GT_vids_takes if takes2use[1] in f.name]
    GT_angrys = [f for f in GT_vids_takes if takes2use[2] in f.name]
    GT_sads = [f for f in GT_vids_takes if takes2use[3] in f.name]
    GT_contempts = [f for f in GT_vids_takes if takes2use[4] in f.name]
    GT_surprises = [f for f in GT_vids_takes if takes2use[5] in f.name]
    GT_feareds = [f for f in GT_vids_takes if takes2use[6] in f.name]
    GT_disgusteds = [f for f in GT_vids_takes if takes2use[7] in f.name]

    GT_neutrals = sorted(GT_neutrals, key=lambda x: int(x.name.split("_")[-2]))
    GT_happys = sorted(GT_happys, key=lambda x: int(x.name.split("_")[-2]))
    GT_angrys = sorted(GT_angrys, key=lambda x: int(x.name.split("_")[-2]))
    GT_sads = sorted(GT_sads, key=lambda x: int(x.name.split("_")[-2]))
    GT_contempts = sorted(GT_contempts, key=lambda x: int(x.name.split("_")[-2]))
    GT_surprises = sorted(GT_surprises, key=lambda x: int(x.name.split("_")[-2]))
    GT_feareds = sorted(GT_feareds, key=lambda x: int(x.name.split("_")[-2]))
    GT_disgusteds = sorted(GT_disgusteds, key=lambda x: int(x.name.split("_")[-2]))

    time_now = time.strftime("%Y%m%d-%H%M%S")
    dump_dir = recons.parent / f"EC_concats_{subject}_{which_take}_{time_now}"
    dump_dir.mkdir(parents=True, exist_ok=True)

    # three_vids, four_vids = True, False
    three_vids, four_vids = False, True

    for i, emo in enumerate(["happys", "angrys", "sads", "contempts", "surprises", "feareds", "disgusteds"]): # for compare against neutral
    # for i, emo in enumerate(["happys", "angrys", "sads", "surprises", "feareds", "disgusteds"]): # for compare against contemtp

        for j in range(min_dirs):
            vid_name = f"{subject}_{takes2use[i]}_{emo}_{j}.mp4"
            vid_name_w_audio = f"{subject}_{takes2use[i]}_{emo}_{j}_audio.mp4"
            
            # # Old Code
            # R1_C1 = GT_neutrals[j]
            # R1_C2 = neutral_recons_vids[j]
            # R2_C1 = eval(f"GT_{emo}[j]")
            # R2_C2 = eval(f"{emo[:-1]}_recons_vids[j]")
            
            if Path(GT_neutrals[j]).exists(): 
                three_vids, four_vids = False, True
                R1_C1 = GT_neutrals[j]
            else: three_vids, four_vids = True, False; R1_C1 = None
            if Path(neutral_recons_vids[j]).exists(): R1_C2 = neutral_recons_vids[j]
            else: continue
            try: R2_C1 = eval(f"GT_{emo}[j]")
            except: print(f"Skipping: {emo} due to no GT"); print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"); continue
            try: R2_C2 = eval(f"{emo[:-1]}_recons_vids[j]")
            except: print(f"Skipping: {emo} due to no recons"); print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"); continue

            if three_vids and not four_vids:
                _ = subprocess.call([
                    "ffmpeg", "-i", str(R1_C2), "-i", str(R2_C1), "-i", str(R2_C2), "-filter_complex", "[0:v][1:v][2:v]hstack=3,format=yuv420p[v]", "-map", "[v]", str(dump_dir / vid_name)
                ])
                _ = subprocess.call([
                    "ffmpeg", "-i", str(dump_dir / vid_name), "-i", str(R1_C2), "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0", "-shortest", str(dump_dir / vid_name_w_audio)
                ])
            
            elif four_vids and not three_vids:
                _ = subprocess.call([
                    "ffmpeg", "-i", str(R1_C1), "-i", str(R1_C2), "-i", str(R2_C1), "-i", str(R2_C2), "-filter_complex", "[0:v][1:v]hstack[top];[2:v][3:v]hstack[bottom];[top][bottom]vstack,format=yuv420p[v]", "-map", "[v]", str(dump_dir / vid_name)
                ])
                _ = subprocess.call([
                    "ffmpeg", "-i", str(dump_dir / vid_name), "-i", str(R1_C2), "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0", "-shortest", str(dump_dir / vid_name_w_audio)
                ])
            
            (dump_dir / vid_name).unlink()
            
    print("DONE!")
            
            

            
