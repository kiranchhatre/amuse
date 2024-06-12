
import io
import sys
import time
import pickle
import random
import string
import torch
import scipy
import librosa
import torchaudio
import numpy as np
from tqdm import tqdm
import torchaudio.transforms as T
from pathlib import Path
from dtw import accelerated_dtw
from numpy.linalg import norm

def audio2mfcc(mfcc_transform, config, all_data, stage, wav_file=None, save_wf=False, version="", ldm_eval=False):
    
    mfcc_per_frame = config["DATA_PARAM"]["Wav"]["mfcc_feat_per_frame"]
    if stage in ["wav_mfcc", "diffusion", "latent_diffusion"]:
        if not ldm_eval:
            actor = wav_file.split("/")[-1].split("_")[1]
            take = "_".join(wav_file.split("/")[-1].split("_")[2:]).split(".")[0]
            SPEECH_WAVEFORM, SAMPLE_RATE = torchaudio.load(wav_file, normalize=True, channels_first=True) # torch.Size([1, x]) 16000
        else: 
            actor, take = "XXX", "0_X_X"
            wav_file = np.array(wav_file.get_array_of_samples())
            wav_file = torch.from_numpy(wav_file).unsqueeze(0).to(torch.float32)
            SPEECH_WAVEFORM, SAMPLE_RATE = wav_file, config["DATA_PARAM"]["Wav"]["sample_rate"]
        
        raw_wf_unmodified = torch.empty_like(SPEECH_WAVEFORM).copy_(SPEECH_WAVEFORM)
        if raw_wf_unmodified.is_cuda: raw_wf_unmodified = raw_wf_unmodified.cpu().numpy()
        else: raw_wf_unmodified = raw_wf_unmodified.numpy()
        
        if stage == "wav_mfcc": 
            SPEECH_WAVEFORM = torch.cat((SPEECH_WAVEFORM, torch.zeros(1, 1920)), dim=1) # padding 0.24 seconds why? TODO: check reason, to avoiding loosing mfcc features 
            SPEECH_WAVEFORM = torch.cat((torch.zeros(1, 1920), SPEECH_WAVEFORM), dim=1) 
        assert SAMPLE_RATE == config["DATA_PARAM"]["Wav"]["sample_rate"], "Sample rate mismatch"
        mfcc = mfcc_transform(SPEECH_WAVEFORM) # (channel, n_mels, time) torch.Size([1, 13, 5906])
        time_len = mfcc.shape[2]
        if not save_wf: all_data = _process_mfcc(actor, take, mfcc, time_len, stage, all_data, mfcc_per_frame)
        else: all_data = _process_mfcc(actor, take, mfcc, time_len, stage, all_data, mfcc_per_frame, raw_wf=raw_wf_unmodified)
                
    elif stage == "wav_dtw_mfcc":
        for i in tqdm(all_data.keys(), desc="[DTW] Processing audio2mfcc...", leave=False):
            for j in all_data[i].keys():
                if version == "v1":
                    if "wav_dtw" in all_data[i][j]:
                        mfcc = all_data[i][j]["wav_dtw"]
                        all_data = _process_mfcc(i, j, mfcc, mfcc.shape[2], stage, all_data, mfcc_per_frame)
                else:
                    mfcc = all_data[i][j]["wav_dtw"]
                    all_data = _process_mfcc(i, j, mfcc, mfcc.shape[2], stage, all_data, mfcc_per_frame)
                 
    else:
        raise ValueError("Invalid audio2mfcc stage!")

    return all_data

def _process_mfcc(actor, take, mfcc, time_len, stage, all_data, mfcc_per_frame, raw_wf=None):
    # clip-wide prediction per frame of 25 fps bvh
    # clip wide audio specifics:
    # original audio stream: 16000 Hz
    # window size = 0.01s
    # remove first coefficient from mfcc [13, 28] -> [12, 28]
    # each audio chunk per frame of bvh: [12, 28]
    
    # More info:
    # https://stackoverflow.com/questions/60474074/preparing-mfcc-audio-feature-should-all-wav-files-be-at-same-length
    # https://stackoverflow.com/questions/54221079/how-to-handle-difference-in-mfcc-feature-for-difference-audio-file
    
    stage = "diff_mfcc" if stage == "diffusion" else stage
    all_data[actor][take][stage] = []
    for i in range(int((time_len-mfcc_per_frame)/4)+1):
        mfcc_sliced = mfcc[:, :, 4 * i : 4 * i + mfcc_per_frame]
        all_data[actor][take][stage].append(mfcc_sliced) # list of lists of torch.Size([1, 13, 28])
    # concatenate all the mfcc features for "diff_mfcc"
    all_data[actor][take][stage] = torch.cat(all_data[actor][take][stage], dim=0) if stage == "diff_mfcc" else all_data[actor][take][stage]
    if raw_wf is not None: all_data[actor][take]["raw_wf"] = raw_wf
    return all_data

def audio2slicedmfcc(config, save_dir=None, actors=None, save_audio=False,
                     batch_audios=None, selected_audio=None, batch_raw_wf=None):
    
    mfcc_transform = T.MFCC(sample_rate=config["DATA_PARAM"]["Wav"]["sample_rate"], 
                            n_mfcc=config["DATA_PARAM"]["Wav"]["n_mfcc"], 
                            melkwargs={"n_fft": config["DATA_PARAM"]["Wav"]["n_fft"],
                                    "n_mels": config["DATA_PARAM"]["Wav"]["n_mels"],
                                    "hop_length": config["DATA_PARAM"]["Wav"]["hop_length"],
                                    "mel_scale": config["DATA_PARAM"]["Wav"]["mel_scale"],},)
    
    batch_audios_flag = True if batch_audios is not None else False
    batch_raw_wf_flag = True if batch_raw_wf is not None else False
    assert (save_audio, batch_audios_flag, batch_raw_wf_flag).count(True) == 1, "Only one of the save_audio, batch_audios, batch_raw_wf can be True"
    
    if save_audio:                                                              # Data preparation 
        save_dir = Path(save_dir) / "wav_files"
        save_dir.mkdir(parents=True, exist_ok=True)
        for actor, arr in zip(actors, selected_audio):
            x = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(5))
            save_file = save_dir / f"{actor}_{x}.wav"
            np_audio = torch.empty_like(arr).copy_(arr).cpu().numpy()
            wav_io = io.BytesIO()
            scipy.io.wavfile.write(wav_io, 16000, np_audio)
            wav_io.seek(0)
            save_file.write_bytes(wav_io.getbuffer())
    elif batch_audios is not None:                                              # Training base AE                
        mfccs = []      
        for wav in batch_audios:
            SPEECH_WAVEFORM, _ = torchaudio.load(wav, normalize=True, channels_first=True)
            mfcc = mfcc_transform(SPEECH_WAVEFORM)
            mfccs.append(mfcc)
        mfccs = torch.stack(mfccs, dim=0)
        return mfccs
    elif batch_raw_wf is not None:                                              # Training baseline ldm        
        batch_raw_wf = batch_raw_wf.permute(0, 2, 1)
        mfcc_transform = mfcc_transform.to(batch_raw_wf.device)
        return mfcc_transform(batch_raw_wf)

def audio2dtw(mfcc_transform, emotion, config, save_path):
    dtw_proc_start = time.time()
    s_rate = config["DATA_PARAM"]["Wav"]["sample_rate"]
    first_wave = True
    for wav_file in emotion.values():
        actor = wav_file.split("/")[-1].split("_")[1]
        take = "_".join(wav_file.split("/")[-1].split("_")[2:]).split(".")[0]
        if first_wave:
            SPEECH_WAVEFORM, SAMPLE_RATE = torchaudio.load(wav_file, normalize=True, channels_first=True)
            SPEECH_WAVEFORM = torch.cat((SPEECH_WAVEFORM, torch.zeros(1, 1920)), dim=1) 
            SPEECH_WAVEFORM = torch.cat((torch.zeros(1, 1920), SPEECH_WAVEFORM), dim=1)
            assert SAMPLE_RATE == s_rate, "Sample rate mismatch"
            mfcc1 = mfcc_transform(SPEECH_WAVEFORM)
            with open(str(save_path) + f"/{actor}_{take}.pkl", 'wb') as f:
                pickle.dump(mfcc1, f)
            first_wave = False
        else:
            SPEECH_WAVEFORM, SAMPLE_RATE = torchaudio.load(wav_file, normalize=True, channels_first=True)
            SPEECH_WAVEFORM = torch.cat((SPEECH_WAVEFORM, torch.zeros(1, 1920)), dim=1)
            SPEECH_WAVEFORM = torch.cat((torch.zeros(1, 1920), SPEECH_WAVEFORM), dim=1)
            assert SAMPLE_RATE == s_rate, "Sample rate mismatch"
            mfcc2 = mfcc_transform(SPEECH_WAVEFORM)
            m1 = mfcc1.permute(0,2,1).squeeze()
            m2 = mfcc2.permute(0,2,1).squeeze()           
            dist, cost, acc_cost, path = accelerated_dtw(m2, m1, dist=lambda x, y: norm(x - y, ord=1))
            # dist, cost.shape, acc_cost.shape,[len(a) for a in path]
            # (1564681.637665987, (8531, 9781), (8531, 9781), [11122, 11122], 533.2730898857117 sec)
            m2_n=m1
            a=path[0]
            b=path[1]
            for l in range(1,len(path[0])):
                m2_n[b[l]] = m2[a[l]]
            with open(str(save_path) + f"/{actor}_{take}.pkl", 'wb') as f:
                    pickle.dump(m2_n.unsqueeze(0).permute(0,2,1), f)
    dtw_proc_ends = time.time()
    # print("Emotion DTW processed, total time: ", dtw_proc_ends - dtw_proc_start) # 3.5 days for 10.5 hours of audio
    
    # DTW info:
    # clip-wide len(mfcc of dtw) for emo: n h a s c s f d: 
    # 2189, 1470, 1533, 1689, 1970, 1376, 1720, 1689 each of torch.Size([1, 13, 28])
    # wav-dtw for above emo shape: 
    # torch.Size([1, 13, 8781]), -> 70.25
    # torch.Size([1, 13, 5906]), -> 47.25
    # torch.Size([1, 13, 6156]), -> 49.25
    # torch.Size([1, 13, 6781]), -> 54.25
    # torch.Size([1, 13, 7906]), -> 63.25
    # torch.Size([1, 13, 5531]), -> 44.25
    # torch.Size([1, 13, 6906]), -> 55.25 
    # torch.Size([1, 13, 6781])  -> 54.25

def audio2dtwdatafix():
    actors = ["yingqing", "goto"] # 27, 25
    takes = ['0_65_65', '0_66_66','0_73_73', '0_74_74',
            '0_81_81', '0_82_82', '0_87_87', '0_88_88', '0_95_95',
            '0_96_96', '0_103_103', '0_104_104', '0_111_111', '0_112_112']
    # CAUTION: fixing only non-neutral takes
    
    for i in tqdm(actors, desc="Actors", leave=False):
        for j in tqdm(takes, desc="Takes", leave=False):
            # ZHAO taken as source for all
            source = "/home/kchhatre/Work/code/disentangled-s2g/data/BEAT-processed/processed-all-modalities/aligned-dtw/zhao_0_"
            
            if j == '0_65_65' or j == '0_66_66':
                source = source + "65_65.pkl"  # DTW aligned MFCC
            elif j == '0_73_73' or j == '0_74_74':
                source = source + "73_73.pkl"
            elif j == '0_81_81' or j == '0_82_82':
                source = source + "81_81.pkl"
            elif j == '0_87_87' or j == '0_88_88':
                source = source + "87_87.pkl"
            elif j == '0_95_95' or j == '0_96_96':
                source = source + "95_95.pkl"
            elif j == '0_103_103' or j == '0_104_104':
                source = source + "103_103.pkl"
            elif j == '0_111_111' or j == '0_112_112':
                source = source + "111_111.pkl"

            with open(source, 'rb') as f:
                zhao_mfcc = pickle.load(f) 
            
            a_index = "27" if i == "yingqing" else "25"
            wave_file = "/home/kchhatre/Work/code/disentangled-s2g/data/beat-rawdata-eng/beat_rawdata_english/"
            wave_file = wave_file + a_index + "/" + a_index + "_" + i + "_" + j + ".wav"
            prealigned_mfcc, _ = torchaudio.load(wave_file, normalize=True, channels_first=True) 
            prealigned_mfcc = torch.cat((prealigned_mfcc, torch.zeros(1, 1920)), dim=1)
            prealigned_mfcc = torch.cat((torch.zeros(1, 1920), prealigned_mfcc), dim=1)
            
            # lazy implementation
            lazy_mfcc_transform = T.MFCC(sample_rate=16000, 
                                n_mfcc=13, 
                                melkwargs={"n_fft": 2048,
                                        "n_mels": 24,
                                        "hop_length": 128,
                                        "mel_scale": "htk",},)
            prealigned_mfcc = lazy_mfcc_transform(prealigned_mfcc)
            
            save_path = "/home/kchhatre/Work/code/disentangled-s2g/data/BEAT-processed/processed-all-modalities/aligned-dtw" + f"/{i}_{j}" + ".pkl"
            
            m1 = zhao_mfcc.permute(0,2,1).squeeze()
            m2 = prealigned_mfcc.permute(0,2,1).squeeze()           
            _, _, _, path = accelerated_dtw(m2, m1, dist=lambda x, y: norm(x - y, ord=1))
            m2_n=m1
            a=path[0]
            b=path[1]
            for l in range(1,len(path[0])):
                m2_n[b[l]] = m2[a[l]]
            with open(save_path, 'wb') as f:
                    pickle.dump(m2_n.unsqueeze(0).permute(0,2,1), f)
                    
def _debug_mfcc_slicing(mfcc):
    mfcc_per_frame = 28
    time_len = mfcc.shape[2]
    debug_all_data = []
    for i in range(int((time_len-mfcc_per_frame)/4)+1):
        mfcc_sliced = mfcc[:, :, 4 * i : 4 * i + mfcc_per_frame]
        debug_all_data.append(mfcc_sliced) 
    debug_all_data = torch.cat(debug_all_data, dim=0) 
    return debug_all_data
            
if __name__ == "__main__":
    
    # Part 1: to fix future dtw misalignments:   
    # audio2dtwdatafix()
    
    # Part 2: DEBUG
    
    src = "/home/kchhatre/Work/code/disentangled-s2g/data/beat-rawdata-eng/beat_rawdata_english/1/1_wayne_0_10_10.wav"
    src1 = "/home/kchhatre/Work/code/disentangled-s2g/data/beat-rawdata-eng/beat_rawdata_english/3/3_solomon_0_11_11.wav"
    subject = "3_solomon_0_11_11"
    dtw = "/home/kchhatre/Work/code/disentangled-s2g/data/BEAT-processed/processed-all-modalities/aligned-dtw/wayne_0_10_10.pkl"

    div_th, total_div = 1, 3
    cut_len = 30 # seconds

    SPEECH_WAVEFORM, SAMPLE_RATE = torchaudio.load(src, normalize=True, channels_first=True)
    print(SPEECH_WAVEFORM.shape, SAMPLE_RATE) # torch.Size([1, 960000]) 16000 | torch.Size([1, 1200000]) 16000
    print("time: ", SPEECH_WAVEFORM.shape[1]/SAMPLE_RATE) # time:  60.0 | time:  75.0
    
    debug_mfcc_transform = T.MFCC(sample_rate=16000, 
                                n_mfcc=13, 
                                melkwargs={"n_fft": 2048,
                                        "n_mels": 128,
                                        "hop_length": 128,
                                        "mel_scale": "htk",},)
    mfcc = debug_mfcc_transform(SPEECH_WAVEFORM) 
    print("src mfcc: ", mfcc.shape) # torch.Size([1, 13, 7501]) | src mfcc:  torch.Size([1, 13, 9376])
    
    # with open(dtw, 'rb') as f:
    #     dtw_mfcc = pickle.load(f)
    # print(dtw_mfcc.shape) # torch.Size([1, 13, 8781])
    # raise Exception("Debugging")
    
    lib_mfcc = librosa.feature.mfcc(SPEECH_WAVEFORM.squeeze().numpy(), sr=SAMPLE_RATE, n_mfcc=13, n_fft=2048, hop_length=128, n_mels=24, htk=True)
    print("lib mfcc: ", lib_mfcc.shape) # lib mfcc:  (13, 9376)
    
    # inverse mfcc using librosa
    mfcc_inv = librosa.feature.inverse.mfcc_to_audio(lib_mfcc, sr=SAMPLE_RATE, n_fft=2048, hop_length=128, htk=True)
    print("lib mfcc inv: ", mfcc_inv.shape) # lib mfcc inv:  (1200000,)
    audio_name = Path("/home/kchhatre/Work/code/disentangled-s2g/viz_dump/diffusion_/audio") / f"{subject}_lib_mfcc_inv.wav"
    # librosa.output.write_wav(audio_name, mfcc_inv, sr=SAMPLE_RATE)
    import soundfile as sf
    sf.write(audio_name, mfcc_inv, SAMPLE_RATE) 
    
    # torch to inv mfcc using librosa
    torchbased_inv_mfcc = librosa.feature.inverse.mfcc_to_audio(mfcc.squeeze().numpy(), sr=SAMPLE_RATE, n_fft=2048, hop_length=128, htk=True)
    print("torch mfcc inv: ", torchbased_inv_mfcc.shape) # torch mfcc inv:  (1200000,)
    audio_name = Path("/home/kchhatre/Work/code/disentangled-s2g/viz_dump/diffusion_/audio") / f"{subject}_torch_mfcc_inv.wav"
    torchaudio.save(audio_name, torch.tensor(torchbased_inv_mfcc).unsqueeze(0), sample_rate=SAMPLE_RATE)
    
    
    """
    # mfcc to wav debug for generation
    mfcc_per_frame = 28
    mfcc_sliced = []
    for i in range(int((mfcc.shape[2]-mfcc_per_frame)/4)+1):
        mfcc_sliced.append(mfcc[:, :, 4 * i : 4 * i + mfcc_per_frame])
    mfcc_sliced = torch.cat(mfcc_sliced, dim=0)
    print(f"mfcc sliced: {mfcc_sliced.shape}") # mfcc sliced: torch.Size([2338, 13, 28])
    waveform_div_factor = SPEECH_WAVEFORM.shape[1] / 10000
    approx_mfcc_time = (waveform_div_factor * 79) - waveform_div_factor + 1
    approx_sliced_mfcc = approx_mfcc_time / 4.01
    start = (int(div_th) * 100) + 1
    end = (int(div_th) + 1) * 100
    approx_wav_start = 10000 * ((4.01 * start) - 1) / 78
    approx_wav_end1 = approx_wav_start + (16000 * cut_len)
    cut_waveform = SPEECH_WAVEFORM[:, int(approx_wav_start):int(approx_wav_end1)]
    aname = f"{div_th}_{total_div}_{subject}"
    audio_name = Path("/home/kchhatre/Work/code/disentangled-s2g/viz_dump/diffusion_/audio") / f"{aname}.wav"   
    torchaudio.save(str(audio_name), cut_waveform, sample_rate=SAMPLE_RATE)
             
    
    
    with open(dtw, 'rb') as f:
        dtw_mfcc = pickle.load(f)
    print(dtw_mfcc.shape) # torch.Size([1, 13, 8781])
    
    mfcc_slice_list = [mfcc, dtw_mfcc]
    
    for mfcc_arr in mfcc_slice_list:
        sliced = _debug_mfcc_slicing(mfcc=mfcc_arr)
        print("sliced mfcc: ", sliced.shape)
        # dtw torch.Size([2189, 13, 28])
        # source torch.Size([1869, 13, 28])
        # diff 2189 - 1869 = 320
    
    # PART 3: TODO: decode audio to wav eg. librosa.feature.inverse.mfcc_to_audio"""