
import math
import json
import copy
import torch
import numpy as np

# Facial Action Coding system 
audio_lip_sync_facs_v0 = ['jawOpen', 'jawForward', 'mouthFunnel', 'mouthPucker', 
                       'mouthLeft', 'mouthRight', 'mouthRollLower', 'mouthRollUpper', 
                       'mouthShrugLower', 'mouthShrugUpper', 'mouthClose', 'mouthUpperUpLeft', 
                       'mouthUpperUpRight', 'mouthLowerDownLeft', 'mouthLowerDownRight', 'mouthPressLeft', 
                       'mouthPressRight', 'mouthStretchLeft', 'mouthStretchRight']
remaining_facs_v0 = ['browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 
                  'browOuterUpRight', 'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight', 
                  'eyeBlinkLeft', 'eyeBlinkRight', 'eyeLookDownLeft', 'eyeLookDownRight', 
                  'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft', 'eyeLookOutRight', 
                  'eyeLookUpLeft', 'eyeLookUpRight', 'eyeSquintLeft', 'eyeSquintRight', 
                  'eyeWideLeft', 'eyeWideRight', 'jawLeft', 'jawRight', 
                  'mouthDimpleLeft', 'mouthDimpleRight', 'mouthFrownLeft', 'mouthFrownRight', 
                  'mouthSmileLeft', 'mouthSmileRight', 'noseSneerLeft', 'noseSneerRight']
combined_facs_v0 = ['browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 
                    'browOuterUpRight', 'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight', 
                    'eyeBlinkLeft', 'eyeBlinkRight', 'eyeLookDownLeft', 'eyeLookDownRight', 
                    'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft', 'eyeLookOutRight', 
                    'eyeLookUpLeft', 'eyeLookUpRight', 'eyeSquintLeft', 'eyeSquintRight', 
                    'eyeWideLeft', 'eyeWideRight', 'jawForward', 'jawLeft', 'jawOpen', 
                    'jawRight', 'mouthClose', 'mouthDimpleLeft', 'mouthDimpleRight', 
                    'mouthFrownLeft', 'mouthFrownRight', 'mouthFunnel', 'mouthLeft', 
                    'mouthLowerDownLeft', 'mouthLowerDownRight', 'mouthPressLeft', 
                    'mouthPressRight', 'mouthPucker', 'mouthRight', 'mouthRollLower', 
                    'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper', 'mouthSmileLeft', 
                    'mouthSmileRight', 'mouthStretchLeft', 'mouthStretchRight', 'mouthUpperUpLeft', 
                    'mouthUpperUpRight', 'noseSneerLeft', 'noseSneerRight']

f_list_dict = {
        "audio_lip_sync_facs_v0": audio_lip_sync_facs_v0,
        "remaining_facs_v0": remaining_facs_v0,
        "combined_facs_v0": combined_facs_v0
}

def process_facial(facial_data, config, l_facs, r_facs):
    
    facial_fps = config["DATA_PARAM"]["Json"]["fps"]
    l_facial, r_facial, facial = [], [], []
    facial_frame_rate = 1/((facial_data["frames"][20]["time"] - facial_data["frames"][10]["time"])/10) # 60 fps
    facial_factor = math.ceil(facial_frame_rate) // facial_fps # 2 for 25 fps
    
    for j, frame_data in enumerate(facial_data["frames"]):
        if j % facial_factor == 0:
            l_facial.append([frame_data["weights"][facial_data["names"].index(fac)] for fac in l_facs])
            r_facial.append([frame_data["weights"][facial_data["names"].index(fac)] for fac in r_facs])
            facial.append(frame_data["weights"])
    l_facial = torch.tensor(np.stack(l_facial))
    r_facial = torch.tensor(np.stack(r_facial)) 
    facial = torch.tensor(np.stack(facial)) 
    return l_facial, r_facial, facial        
   
def faces_con_emo_combine(face1, face2, source_face_file, disentangled, save_path, filename, cfg, verbose=False):  
    lip_bsw = f_list_dict[cfg["DATA_PARAM"]["Json"]["con_emo_div"]["con"]]      
    nonlip_bsw = f_list_dict[cfg["DATA_PARAM"]["Json"]["con_emo_div"]["emo"]] 
    combined_bsw = f_list_dict[cfg["DATA_PARAM"]["Json"]["con_emo_div"]["combined"]] 
    
    frame1, feat1 = face1.shape
    if face2 is not None:
        frame2, feat2 = face2.shape 
        assert frame1 == frame2, "[DIFF JSON COMBINE] Number of frames are not same"
        assert disentangled, "[DIFF JSON COMBINE] Disentangled should be True if you are combining two different json files"
    
    with open(source_face_file) as f:
        source_face = json.load(f)
    combined_json = copy.deepcopy(source_face)
    combined_json["frames"] = combined_json["frames"][:frame1]
    
    combined = []
    for i in range(frame1):
        combo_tmp = dict((x,0) for x in combined_bsw) # initialize with 0
        if face2 is not None:
            lip_bsw_tmp = dict((y, 0) for y in lip_bsw)
            lip_bsw_tmp.update(dict(zip(lip_bsw, face1[i].tolist())))
            nonlip_bsw_tmp = dict((z, 0) for z in nonlip_bsw)
            nonlip_bsw_tmp.update(dict(zip(nonlip_bsw, face2[i].tolist())))    
            combo_tmp.update(lip_bsw_tmp)
            combo_tmp.update(nonlip_bsw_tmp)
        else:
            combo_tmp.update(dict(zip(combined_bsw, face1[i].tolist())))
        combined.append(list(combo_tmp.values()))
        assert len(combined[i]) == len(combined_bsw), "[DIFF JSON COMBINE] Combined facial features (per frame) are not same as combined_bsw"
    
    combined = torch.tensor(np.stack(combined))
    assert combined.shape == (frame1, len(combined_bsw)), "[DIFF JSON COMBINE] Combined facial features are not same as combined_bsw"
    
    # write to json
    facial_fps = cfg["DATA_PARAM"]["Json"]["fps"]
    for i in range(frame1):
        combined_json["frames"][i]["time"] = (i+1)/facial_fps
        combined_json["frames"][i]["weights"] = combined[i].tolist()
    
    filename = f"{save_path / filename}.json"
    with open(str(filename), "w") as f:
        json.dump(combined_json, f)
    return filename

def std_faces_forward_backward(facial_data, config, forward=True):      # TODO: implement standardization using sklearn StandardScaler
    raise NotImplementedError(f"preprocess_faces_forward_backward not implemented yet")
             
if __name__ == "__main__":
    
    # DEBUG
    
    d_audio_lip_sync_facs = ['jawOpen', 'jawForward', 'mouthFunnel', 'mouthPucker', 
                                    'mouthLeft', 'mouthRight', 'mouthRollLower', 'mouthRollUpper', 
                                    'mouthShrugLower', 'mouthShrugUpper', 'mouthClose', 'mouthUpperUpLeft', 
                                    'mouthUpperUpRight', 'mouthLowerDownLeft', 'mouthLowerDownRight', 'mouthPressLeft', 
                                    'mouthPressRight', 'mouthStretchLeft', 'mouthStretchRight']
    d_remaining_facs = ['browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 
                            'browOuterUpRight', 'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight', 
                            'eyeBlinkLeft', 'eyeBlinkRight', 'eyeLookDownLeft', 'eyeLookDownRight', 
                            'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft', 'eyeLookOutRight', 
                            'eyeLookUpLeft', 'eyeLookUpRight', 'eyeSquintLeft', 'eyeSquintRight', 
                            'eyeWideLeft', 'eyeWideRight', 'jawLeft', 'jawRight', 
                            'mouthDimpleLeft', 'mouthDimpleRight', 'mouthFrownLeft', 'mouthFrownRight', 
                            'mouthSmileLeft', 'mouthSmileRight', 'noseSneerLeft', 'noseSneerRight']
    
    f_base_path = "/home/kchhatre/Work/code/disentangled-s2g/data/beat-rawdata-eng/beat_rawdata_english/1"
    facial_file = f_base_path + "/1_wayne_0_10_10.json"
    
    b_base_path = "/home/kchhatre/Work/code/disentangled-s2g/data/BEAT-processed/processed-all-modalities/bvh-fps25-joint_name_list_27-div-f4-v0"
    bvh_file = b_base_path + "/1_wayne_0_10_10_con.bvh"
    
    all_poses = []
    with open(bvh_file, "r") as f:
        for j, line in enumerate(f.readlines()):
            pose_data = np.fromstring(line, dtype=float, sep=" ")
            all_poses.append(pose_data)
    all_poses = np.array(all_poses)
    bvh_frames = all_poses.shape[0] 
    debug_config = {
        "DATA_PARAM": {
            "Json": {
                "fps": 25
            }
        }
    }
    
    with open(facial_file, "r") as f:
        facial_data = json.load(f)
    
    # facial_each_file = process_facial(facial_data, bvh_frames, fps)
    d_l_facial, d_r_facial, d_facial = process_facial(facial_data, debug_config, d_audio_lip_sync_facs, d_remaining_facs)
    print(d_l_facial[0])
    print(d_r_facial[0])
    print(d_l_facial.shape, d_r_facial.shape, d_facial.shape)