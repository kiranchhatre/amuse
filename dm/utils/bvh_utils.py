

import sys
# import bvhio                                                                  # buggy
import pickle
import collections
import math
import glob
import torch
import numpy as np
from bvh import Bvh
import joblib as jl
from pathlib import Path
from tqdm import tqdm
from einops import rearrange
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from dm.utils.PyMO.pymo.parsers import BVHParser
from dm.utils.PyMO.pymo.viz_tools import *
from dm.utils.PyMO.pymo.preprocessing import *
from dm.utils.PyMO.pymo.features import *
from dm.utils.PyMO.pymo.writers import *
from dm.utils.sk2torch import sk2torch
from dm.utils.transforms import *
from dm.utils import bvh

beat_joints = collections.OrderedDict()

# BEAT source ##################################################################

beat_joints={
        'Hips':         [6,6],
        'Spine':        [3,9],
        'Spine1':       [3,12],
        'Spine2':       [3,15],
        'Spine3':       [3,18],
        'Neck':         [3,21],
        'Neck1':        [3,24],
        'Head':         [3,27],
        'HeadEnd':      [3,30],

        'RShoulder':    [3,33], 
        'RArm':         [3,36],
        'RArm1':        [3,39],
        'RHand':        [3,42],    
        'RHandM1':      [3,45],
        'RHandM2':      [3,48],
        'RHandM3':      [3,51],
        'RHandM4':      [3,54],

        'RHandR':       [3,57],
        'RHandR1':      [3,60],
        'RHandR2':      [3,63],
        'RHandR3':      [3,66],
        'RHandR4':      [3,69],

        'RHandP':       [3,72],
        'RHandP1':      [3,75],
        'RHandP2':      [3,78],
        'RHandP3':      [3,81],
        'RHandP4':      [3,84],

        'RHandI':       [3,87],
        'RHandI1':      [3,90],
        'RHandI2':      [3,93],
        'RHandI3':      [3,96],
        'RHandI4':      [3,99],

        'RHandT1':      [3,102],
        'RHandT2':      [3,105],
        'RHandT3':      [3,108],
        'RHandT4':      [3,111],

        'LShoulder':    [3,114], 
        'LArm':         [3,117],
        'LArm1':        [3,120],
        'LHand':        [3,123],    
        'LHandM1':      [3,126],
        'LHandM2':      [3,129],
        'LHandM3':      [3,132],
        'LHandM4':      [3,135],

        'LHandR':       [3,138],
        'LHandR1':      [3,141],
        'LHandR2':      [3,144],
        'LHandR3':      [3,147],
        'LHandR4':      [3,150],

        'LHandP':       [3,153],
        'LHandP1':      [3,156],
        'LHandP2':      [3,159],
        'LHandP3':      [3,162],
        'LHandP4':      [3,165],

        'LHandI':       [3,168],
        'LHandI1':      [3,171],
        'LHandI2':      [3,174],
        'LHandI3':      [3,177],
        'LHandI4':      [3,180],

        'LHandT1':      [3,183],
        'LHandT2':      [3,186],
        'LHandT3':      [3,189],
        'LHandT4':      [3,192],

        'RUpLeg':       [3,195],
        'RLeg':         [3,198],
        'RFoot':        [3,201],
        'RFootF':       [3,204],
        'RToeBase':     [3,207],
        'RToeBaseEnd':  [3,210],

        'LUpLeg':       [3,213],
        'LLeg':         [3,216],
        'LFoot':        [3,219],
        'LFootF':       [3,222],
        'LToeBase':     [3,225],
        'LToeBaseEnd':  [3,228]}

joint_name_list_225 =  { # J: 75  ALL JOINTS
        'Hips':        3 , 
        'Spine':       3 ,
        'Spine1':      3 ,
        'Spine2':      3 ,
        'Spine3':      3 ,
        'Neck':        3 ,
        'Neck1':       3 ,
        'Head':        3 ,
        'HeadEnd':     3 ,
        'RShoulder':   3 , 
        'RArm':        3 ,
        'RArm1':       3 ,
        'RHand':       3 ,    
        'RHandM1':     3 ,
        'RHandM2':     3 ,
        'RHandM3':     3 ,
        'RHandM4':     3 ,
        'RHandR':      3 ,
        'RHandR1':     3 ,
        'RHandR2':     3 ,
        'RHandR3':     3 ,
        'RHandR4':     3 ,
        'RHandP':      3 ,
        'RHandP1':     3 ,
        'RHandP2':     3 ,
        'RHandP3':     3 ,
        'RHandP4':     3 ,
        'RHandI':      3 ,
        'RHandI1':     3 ,
        'RHandI2':     3 ,
        'RHandI3':     3 ,
        'RHandI4':     3 ,
        'RHandT1':     3 ,
        'RHandT2':     3 ,
        'RHandT3':     3 ,
        'RHandT4':     3 ,
        'LShoulder':   3 , 
        'LArm':        3 ,
        'LArm1':       3 ,
        'LHand':       3 ,    
        'LHandM1':     3 ,
        'LHandM2':     3 ,
        'LHandM3':     3 ,
        'LHandM4':     3 ,
        'LHandR':      3 ,
        'LHandR1':     3 ,
        'LHandR2':     3 ,
        'LHandR3':     3 ,
        'LHandR4':     3 ,
        'LHandP':      3 ,
        'LHandP1':     3 ,
        'LHandP2':     3 ,
        'LHandP3':     3 ,
        'LHandP4':     3 ,
        'LHandI':      3 ,
        'LHandI1':     3 ,
        'LHandI2':     3 ,
        'LHandI3':     3 ,
        'LHandI4':     3 ,
        'LHandT1':     3 ,
        'LHandT2':     3 ,
        'LHandT3':     3 ,
        'LHandT4':     3 ,
        'RUpLeg':      3 ,
        'RLeg':        3 ,
        'RFoot':       3 ,
        'RFootF':      3 ,
        'RToeBase':    3 ,
        'RToeBaseEnd': 3 ,
        'LUpLeg':      3 ,
        'LLeg':        3 ,
        'LFoot':       3 ,
        'LFootF':      3 ,
        'LToeBase':    3 ,
        'LToeBaseEnd': 3 }

joint_name_list_186 =  { # J: 62
        'Hips':        3 , 
        'Spine':       3 ,
        'Spine1':      3 ,
        'Spine2':      3 ,
        'Spine3':      3 ,
        'Neck':        3 ,
        'Neck1':       3 ,
        'Head':        3 ,
        'RShoulder':   3 , 
        'RArm':        3 ,
        'RArm1':       3 ,
        'RHand':       3 ,    
        'RHandM1':     3 ,
        'RHandM2':     3 ,
        'RHandM3':     3 ,
        'RHandR':      3 ,
        'RHandR1':     3 ,
        'RHandR2':     3 ,
        'RHandR3':     3 ,
        'RHandP':      3 ,
        'RHandP1':     3 ,
        'RHandP2':     3 ,
        'RHandP3':     3 ,
        'RHandI':      3 ,
        'RHandI1':     3 ,
        'RHandI2':     3 ,
        'RHandI3':     3 ,
        'RHandT1':     3 ,
        'RHandT2':     3 ,
        'RHandT3':     3 ,
        'LShoulder':   3 , 
        'LArm':        3 ,
        'LArm1':       3 ,
        'LHand':       3 ,    
        'LHandM1':     3 ,
        'LHandM2':     3 ,
        'LHandM3':     3 ,
        'LHandR':      3 ,
        'LHandR1':     3 ,
        'LHandR2':     3 ,
        'LHandR3':     3 ,
        'LHandP':      3 ,
        'LHandP1':     3 ,
        'LHandP2':     3 ,
        'LHandP3':     3 ,
        'LHandI':      3 ,
        'LHandI1':     3 ,
        'LHandI2':     3 ,
        'LHandI3':     3 ,
        'LHandT1':     3 ,
        'LHandT2':     3 ,
        'LHandT3':     3 ,
        'RUpLeg':      3 ,
        'RLeg':        3 ,
        'RFoot':       3 ,
        'RFootF':      3 ,
        'RToeBase':    3 ,
        'LUpLeg':      3 ,
        'LLeg':        3 ,
        'LFoot':       3 ,
        'LFootF':      3 ,
        'LToeBase':    3 }

joint_name_list_27 =  { # J: 9
        'Hips':        3 , 
        'Neck':        3 ,
        'Head':        3 ,
        'RShoulder':   3 , 
        'RArm':        3 ,
        'RArm1':       3 ,    
        'LShoulder':   3 , 
        'LArm':        3 ,
        'LArm1':       3 }   

joint_name_list_27_v3 =  {  # J: 9
        'Spine3':      3 , 
        'Neck':        3 ,
        'Head':        3 ,
        'RShoulder':   3 , 
        'RArm':        3 ,
        'RArm1':       3 ,    
        'LShoulder':   3 , 
        'LArm':        3 ,
        'LArm1':       3 } 

spine_neck_141 =  {  # J: 47
        'Spine':       3 , # CaMN non-hands
        'Neck':        3 , # CaMN non-hands
        'Neck1':       3 , # CaMN non-hands
        'RShoulder':   3 , # CaMN non-hands
        'RArm':        3 , # CaMN non-hands
        'RArm1':       3 , # CaMN non-hands
        'RHand':       3 ,    
        'RHandM1':     3 ,
        'RHandM2':     3 ,
        'RHandM3':     3 ,
        'RHandR':      3 ,
        'RHandR1':     3 ,
        'RHandR2':     3 ,
        'RHandR3':     3 ,
        'RHandP':      3 ,
        'RHandP1':     3 ,
        'RHandP2':     3 ,
        'RHandP3':     3 ,
        'RHandI':      3 ,
        'RHandI1':     3 ,
        'RHandI2':     3 ,
        'RHandI3':     3 ,
        'RHandT1':     3 ,
        'RHandT2':     3 ,
        'RHandT3':     3 ,
        'LShoulder':   3 , # CaMN non-hands
        'LArm':        3 , # CaMN non-hands
        'LArm1':       3 , # CaMN non-hands
        'LHand':       3 ,    
        'LHandM1':     3 ,
        'LHandM2':     3 ,
        'LHandM3':     3 ,
        'LHandR':      3 ,
        'LHandR1':     3 ,
        'LHandR2':     3 ,
        'LHandR3':     3 ,
        'LHandP':      3 ,
        'LHandP1':     3 ,
        'LHandP2':     3 ,
        'LHandP3':     3 ,
        'LHandI':      3 ,
        'LHandI1':     3 ,
        'LHandI2':     3 ,
        'LHandI3':     3 ,
        'LHandT1':     3 ,
        'LHandT2':     3 ,
        'LHandT3':     3 }

remaining_j_v0 =  {  # J: 10
        'Hips':        3 ,
        'Neck':        3 ,
        'Head':        3 ,
        'Spine1':      3 , 
        'RUpLeg':      3 ,
        'RLeg':        3 ,
        'RFoot':       3 ,
        'LUpLeg':      3 ,
        'LLeg':        3 ,
        'LFoot':       3 }

audio_sync_j_v0 = { # J: 8
        'RShoulder':   3 , 
        'RArm':        3 ,
        'RArm1':       3 ,
        'RHand':       3 ,
        'LShoulder':   3 , 
        'LArm':        3 ,
        'LArm1':       3 ,
        'LHand':       3 }

combined_v0 =  {  # J: 18
        'Hips':        3 ,
        'Spine1':      3 , 
        'Neck':        3 ,
        'Head':        3 ,
        'RShoulder':   3 , 
        'RArm':        3 ,
        'RArm1':       3 ,
        'RHand':       3 ,
        'LShoulder':   3 , 
        'LArm':        3 ,
        'LArm1':       3 ,
        'LHand':       3 ,
        'RUpLeg':      3 ,
        'RLeg':        3 ,
        'RFoot':       3 ,
        'LUpLeg':      3 ,
        'LLeg':        3 ,
        'LFoot':       3 }

# SMPL based joint list ########################################################

# Additional source:
# https://learn.microsoft.com/en-us/azure/kinect-dk/body-joints?source=recommendations
KINECT_V2 = {   
   "SpineBase" : 1,
   "SpineMid" : 2,
   "Neck" : 3,
   "Head" : 4,
   "ShoulderLeft" : 5,
   "ElbowLeft" : 6,
   "WristLeft" : 7,
   "HandLeft" : 8,
   "ShoulderRight" : 9,
   "ElbowRight" : 10,
   "WristRight" : 11,
   "HandRight" : 12,
   "HipLeft" : 13,
   "KneeLeft" : 14,
   "AnkleLeft" : 15,
   "FootLeft" : 16, 
   "HipRight" : 17,
   "KneeRight" : 18,
   "AnkleRight" : 19,
   "FootRight" : 20,
   "SpineShoulder" : 21,
   "HandTipLeft" : 22,
   "ThumbLeft" : 23,
   "HandTipRight" : 24,
   "ThumbRight" : 25,
}

# MLD based (assumed Kinect)
JOINT_MAP = {     # Pymo selection
'MidHip': 0,      # Hips (automatically included in PyMO)
'LHip': 1,        # LeftUpLeg
'LKnee': 4,       # LeftLeg
'LAnkle': 7,      # LeftFoot
'LFoot': 10,      # LeftToeBase
'RHip': 2,        # RightUpLeg
'RKnee': 5,       # RightLeg
'RAnkle': 8,      # RightFoot    
'RFoot': 11,      # RightToeBase
'LShoulder': 16,  # LeftArm
'LElbow': 18,     # LeftForeArm
'LWrist': 20,     # LeftHand
'RShoulder': 17,  # RightArm
'RElbow': 19,     # RightForeArm
'RWrist': 21,     # RightHand
'spine1': 3,      # Spine1
'spine2': 6,      # Spine2
'spine3': 9,      # Spine3
'Neck': 12,       # Neck
'Head': 15,       # Head
'LCollar':13,     # LeftShoulder
'Rcollar' :14     # RightShoulder
}

# BEAT joints processed with PyMO mapped to Kinect for AMASS conversion using MLD
BP2KA_old = [  # Hips included
    'LeftUpLeg',
    'LeftLeg',
    'LeftFoot',
    'LeftToeBase',
    'RightUpLeg',
    'RightLeg',
    'RightFoot',
    'RightToeBase',
    'LeftArm',
    'LeftForeArm',
    'LeftHand',
    'RightArm',
    'RightForeArm',
    'RightHand',
    'Spine1',
    'Spine2',
    'Spine3',
    'Neck',
    'Head',
    'LeftShoulder',
    'RightShoulder',
]

BP2KA = [  # Hips included
    'LeftUpLeg',
    'RightUpLeg',
    'Spine1',
    'LeftLeg',
    'RightLeg',
    'Spine2',
    'LeftFoot',
    'RightFoot',
    'Spine3',
    'LeftToeBase',
    'RightToeBase', 
    'Neck',  # Neck
    'LeftShoulder',# LeftShoulder
    'RightShoulder',
    'Head',
    'LeftArm',
    'RightArm',
    'LeftForeArm',
    'RightForeArm',
    'LeftHand',
    'RightHand'
]

BP2KA_feats = [ # 22 * 3 (also with RootCentricPositionNormalizer)
    'Hips_Xposition', 
    'Hips_Yposition', 
    'Hips_Zposition', 
    'LeftUpLeg_Xposition', 
    'LeftUpLeg_Yposition', 
    'LeftUpLeg_Zposition', 
    'LeftLeg_Xposition', 
    'LeftLeg_Yposition', 
    'LeftLeg_Zposition', 
    'LeftFoot_Xposition', 
    'LeftFoot_Yposition', 
    'LeftFoot_Zposition', 
    'LeftToeBase_Xposition', 
    'LeftToeBase_Yposition', 
    'LeftToeBase_Zposition', 
    'RightUpLeg_Xposition', 
    'RightUpLeg_Yposition', 
    'RightUpLeg_Zposition', 
    'RightLeg_Xposition', 
    'RightLeg_Yposition', 
    'RightLeg_Zposition', 
    'RightFoot_Xposition', 
    'RightFoot_Yposition', 
    'RightFoot_Zposition', 
    'RightToeBase_Xposition', 
    'RightToeBase_Yposition', 
    'RightToeBase_Zposition',
    'LeftArm_Xposition', 
    'LeftArm_Yposition', 
    'LeftArm_Zposition', 
    'LeftForeArm_Xposition', 
    'LeftForeArm_Yposition',
    'LeftForeArm_Zposition', 
    'LeftHand_Xposition', 
    'LeftHand_Yposition', 
    'LeftHand_Zposition', 
    'RightArm_Xposition', 
    'RightArm_Yposition', 
    'RightArm_Zposition', 
    'RightForeArm_Xposition', 
    'RightForeArm_Yposition', 
    'RightForeArm_Zposition', 
    'RightHand_Xposition', 
    'RightHand_Yposition',
    'RightHand_Zposition', 
    'Spine1_Xposition', 
    'Spine1_Yposition', 
    'Spine1_Zposition', 
    'Spine2_Xposition', 
    'Spine2_Yposition', 
    'Spine2_Zposition', 
    'Spine3_Xposition', 
    'Spine3_Yposition', 
    'Spine3_Zposition', 
    'Neck_Xposition', 
    'Neck_Yposition',
    'Neck_Zposition',
    'Head_Xposition', 
    'Head_Yposition',
    'Head_Zposition', 
    'LeftShoulder_Xposition',
    'LeftShoulder_Yposition',
    'LeftShoulder_Zposition',
    'RightShoulder_Xposition', 
    'RightShoulder_Yposition', 
    'RightShoulder_Zposition'
]

smpl_skeleton = {
     0: 'pelvis',
     1: 'left_hip',
     2: 'right_hip',
     3: 'spine1',
     4: 'left_knee',
     5: 'right_knee',
     6: 'spine2',
     7: 'left_ankle',
     8: 'right_ankle',
     9: 'spine3',
    10: 'left_foot',
    11: 'right_foot',
    12: 'neck',
    13: 'left_collar',
    14: 'right_collar',
    15: 'head',
    16: 'left_shoulder',
    17: 'right_shoulder',
    18: 'left_elbow',
    19: 'right_elbow',
    20: 'left_wrist',
    21: 'right_wrist',
    22: 'left_hand',
    23: 'right_hand'
}

smplh_skeleton = {
     0: 'pelvis',
     1: 'left_hip',
     2: 'right_hip',
     3: 'spine1',
     4: 'left_knee',
     5: 'right_knee',
     6: 'spine2',
     7: 'left_ankle',
     8: 'right_ankle',
     9: 'spine3',
    10: 'left_foot',
    11: 'right_foot',
    12: 'neck',
    13: 'left_collar',
    14: 'right_collar',
    15: 'head',
    16: 'left_shoulder',
    17: 'right_shoulder',
    18: 'left_elbow',
    19: 'right_elbow',
    20: 'left_wrist',
    21: 'right_wrist',
    22: 'left_index1',
    23: 'left_index2',
    24: 'left_index3',
    25: 'left_middle1',
    26: 'left_middle2',
    27: 'left_middle3',
    28: 'left_pinky1',
    29: 'left_pinky2',
    30: 'left_pinky3',
    31: 'left_ring1',
    32: 'left_ring2',
    33: 'left_ring3',
    34: 'left_thumb1',
    35: 'left_thumb2',
    36: 'left_thumb3',
    37: 'right_index1',
    38: 'right_index2',
    39: 'right_index3',
    40: 'right_middle1',
    41: 'right_middle2',
    42: 'right_middle3',
    43: 'right_pinky1',
    44: 'right_pinky2',
    45: 'right_pinky3',
    46: 'right_ring1',
    47: 'right_ring2',
    48: 'right_ring3',
    49: 'right_thumb1',
    50: 'right_thumb2',
    51: 'right_thumb3'
}

smplx_skeleton = {
     0: 'pelvis',
     1: 'left_hip',
     2: 'right_hip',
     3: 'spine1',
     4: 'left_knee',
     5: 'right_knee',
     6: 'spine2',
     7: 'left_ankle',
     8: 'right_ankle',
     9: 'spine3',
    10: 'left_foot',
    11: 'right_foot',
    12: 'neck',
    13: 'left_collar',
    14: 'right_collar',
    15: 'head',
    16: 'left_shoulder',
    17: 'right_shoulder',
    18: 'left_elbow',
    19: 'right_elbow',
    20: 'left_wrist',
    21: 'right_wrist',
    22: 'jaw',
    23: 'left_eye',
    24: 'right_eye',
    25: 'left_index1',
    26: 'left_index2',
    27: 'left_index3',
    28: 'left_middle1',
    29: 'left_middle2',
    30: 'left_middle3',
    31: 'left_pinky1',
    32: 'left_pinky2',
    33: 'left_pinky3',
    34: 'left_ring1',
    35: 'left_ring2',
    36: 'left_ring3',
    37: 'left_thumb1',
    38: 'left_thumb2',
    39: 'left_thumb3',
    40: 'right_index1',
    41: 'right_index2',
    42: 'right_index3',
    43: 'right_middle1',
    44: 'right_middle2',
    45: 'right_middle3',
    46: 'right_pinky1',
    47: 'right_pinky2',
    48: 'right_pinky3',
    49: 'right_ring1',
    50: 'right_ring2',
    51: 'right_ring3',
    52: 'right_thumb1',
    53: 'right_thumb2',
    54: 'right_thumb3'
}

supr_skeleton = {
    0: 'pelvis',
    1: 'left_hip',
    2: 'right_hip',
    3: 'spine1',
    4: 'left_knee',
    5: 'right_knee',
    6: 'spine2',
    7: 'left_ankle',
    8: 'right_ankle',
    9: 'spine3',
    10: 'left_foot',
    11: 'right_foot',
    12: 'neck',
    13: 'left_collar',
    14: 'right_collar',
    15: 'head',
    16: 'left_shoulder',
    17: 'right_shoulder',
    18: 'left_elbow',
    19: 'right_elbow',
    20: 'left_wrist',
    21: 'right_wrist',
    22: 'jaw',
    23: 'left_eye',
    24: 'right_eye',
    25: 'left_index1',
    26: 'left_index2',
    27: 'left_index3',
    28: 'left_middle1',
    29: 'left_middle2',
    30: 'left_middle3',
    31: 'left_pinky1',
    32: 'left_pinky2',
    33: 'left_pinky3',
    34: 'left_ring1',
    35: 'left_ring2',
    36: 'left_ring3',
    37: 'left_thumb1',
    38: 'left_thumb2',
    39: 'left_thumb3',
    40: 'right_index1',
    41: 'right_index2',
    42: 'right_index3',
    43: 'right_middle1',
    44: 'right_middle2',
    45: 'right_middle3',
    46: 'right_pinky1',
    47: 'right_pinky2',
    48: 'right_pinky3',
    49: 'right_ring1',
    50: 'right_ring2',
    51: 'right_ring3',
    52: 'right_thumb1',
    53: 'right_thumb2',
    54: 'right_thumb3',
    55: 'left_bigtoe1',
    56: 'left_bigtoe2',
    57: 'left_indextoe1',
    58: 'left_indextoe2',
    59: 'left_middletoe1',
    60: 'left_middletoe2',
    61: 'left_ringtoe1',
    62: 'left_ringtoe2',
    63: 'left_pinkytoe1',
    64: 'left_pinkytoe2',
    65: 'right_bigtoe1',
    66: 'right_bigtoe2',
    67: 'right_indextoe1',
    68: 'right_indextoe2',
    69: 'right_middletoe1',
    70: 'right_middletoe2',
    71: 'right_ringtoe1',
    72: 'right_ringtoe2',
    73: 'right_pinkytoe1',
    74: 'right_pinkytoe2'
}

# MotionMatching repo based joint list #########################################

beat_MM = [
    'Hips', 'Spine', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Neck1', 'Head', 'HeadEnd', 
    'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'RightHandMiddle1', 
    'RightHandMiddle2', 'RightHandMiddle3', 'RightHandMiddle4', 'RightHandRing', 
    'RightHandRing1', 'RightHandRing2', 'RightHandRing3', 'RightHandRing4', 'RightHandPinky', 
    'RightHandPinky1', 'RightHandPinky2', 'RightHandPinky3', 'RightHandPinky4', 'RightHandIndex',
    'RightHandIndex1', 'RightHandIndex2', 'RightHandIndex3', 'RightHandIndex4', 'RightHandThumb1',
    'RightHandThumb2', 'RightHandThumb3', 'RightHandThumb4', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 
    'LeftHand', 'LeftHandMiddle1', 'LeftHandMiddle2', 'LeftHandMiddle3', 'LeftHandMiddle4',
    'LeftHandRing', 'LeftHandRing1', 'LeftHandRing2', 'LeftHandRing3', 'LeftHandRing4', 'LeftHandPinky',
    'LeftHandPinky1', 'LeftHandPinky2', 'LeftHandPinky3', 'LeftHandPinky4', 'LeftHandIndex', 'LeftHandIndex1', 
    'LeftHandIndex2', 'LeftHandIndex3', 'LeftHandIndex4', 'LeftHandThumb1', 'LeftHandThumb2', 
    'LeftHandThumb3', 'LeftHandThumb4', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightForeFoot', 
    'RightToeBase', 'RightToeBaseEnd', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftForeFoot', 'LeftToeBase', 'LeftToeBaseEnd']
    
beat2smpl_new_MM = [  
    'Hips', 
    'LeftUpLeg',
    'RightUpLeg',
    'Spine1',
    'LeftLeg',
    'RightLeg',
    'Spine2',
    'LeftFoot',
    'RightFoot',
    'Spine3',
    'LeftToeBase',
    'RightToeBase', 
    'Neck',  # Neck
    'LeftShoulder',# LeftShoulder
    'RightShoulder',
    'Head',
    'LeftArm',
    'RightArm',
    'LeftForeArm',
    'RightForeArm',
    'LeftHand',
    'RightHand']

# PyMO based joint list ########################################################

#  Hips (None)  --[BEAT PRESENT]                                                                     
# | | - LeftUpLeg (Hips)  --[BEAT PRESENT]
# | | - LeftLeg (LeftUpLeg)  --[BEAT PRESENT]
# | | - LeftFoot (LeftLeg)  --[BEAT PRESENT]
# | | - LeftForeFoot (LeftFoot)  --[BEAT PRESENT]
# | | - LeftToeBase (LeftForeFoot)  --[BEAT PRESENT]
# | | - LeftToeBaseEnd (LeftToeBase)  --[BEAT PRESENT]
# | | - LeftToeBaseEnd_Nub (LeftToeBaseEnd)  --[BEAT NOT PRESENT]
# | - RightUpLeg (Hips)  --[BEAT PRESENT]
# | - RightLeg (RightUpLeg)  --[BEAT PRESENT]
# | - RightFoot (RightLeg)  --[BEAT PRESENT]
# | - RightForeFoot (RightFoot)  --[BEAT PRESENT]
# | - RightToeBase (RightForeFoot)  --[BEAT PRESENT]
# | - RightToeBaseEnd (RightToeBase)  --[BEAT PRESENT]
# | - RightToeBaseEnd_Nub (RightToeBaseEnd)  --[BEAT NOT PRESENT]
# - Spine (Hips)  --[BEAT PRESENT]
# - Spine1 (Spine)  --[BEAT PRESENT]
# - Spine2 (Spine1)  --[BEAT PRESENT]
# - Spine3 (Spine2)  --[BEAT PRESENT]
# | | - LeftShoulder (Spine3)  --[BEAT PRESENT]
# | | - LeftArm (LeftShoulder)  --[BEAT PRESENT]
# | | - LeftForeArm (LeftArm)  --[BEAT PRESENT]
# | | - LeftHand (LeftForeArm)  --[BEAT PRESENT]
# | | | | - LeftHandIndex (LeftHand)  --[BEAT PRESENT] [FINGERS]
# | | | | | - LeftHandThumb1 (LeftHandIndex)  --[BEAT PRESENT] [FINGERS]
# | | | | | - LeftHandThumb2 (LeftHandThumb1)  --[BEAT PRESENT] [FINGERS]
# | | | | | - LeftHandThumb3 (LeftHandThumb2)  --[BEAT PRESENT] [FINGERS]
# | | | | | - LeftHandThumb4 (LeftHandThumb3)  --[BEAT PRESENT] [FINGERS]
# | | | | | - LeftHandThumb4_Nub (LeftHandThumb4)  --[BEAT NOT PRESENT]
# | | | | - LeftHandIndex1 (LeftHandIndex)  --[BEAT PRESENT] [FINGERS]
# | | | | - LeftHandIndex2 (LeftHandIndex1)  --[BEAT PRESENT] [FINGERS]
# | | | | - LeftHandIndex3 (LeftHandIndex2)  --[BEAT PRESENT] [FINGERS]
# | | | | - LeftHandIndex4 (LeftHandIndex3)  --[BEAT PRESENT] [FINGERS]
# | | | | - LeftHandIndex4_Nub (LeftHandIndex4)  --[BEAT NOT PRESENT]
# | | | - LeftHandRing (LeftHand)  --[BEAT PRESENT] [FINGERS]
# | | | | - LeftHandPinky (LeftHandRing)  --[BEAT PRESENT] [FINGERS]
# | | | | - LeftHandPinky1 (LeftHandPinky)  --[BEAT PRESENT] [FINGERS]
# | | | | - LeftHandPinky2 (LeftHandPinky1)  --[BEAT PRESENT] [FINGERS]
# | | | | - LeftHandPinky3 (LeftHandPinky2)  --[BEAT PRESENT] [FINGERS]
# | | | | - LeftHandPinky4 (LeftHandPinky3)  --[BEAT PRESENT] [FINGERS]
# | | | | - LeftHandPinky4_Nub (LeftHandPinky4)  --[BEAT NOT PRESENT]
# | | | - LeftHandRing1 (LeftHandRing)  --[BEAT PRESENT] [FINGERS]
# | | | - LeftHandRing2 (LeftHandRing1)  --[BEAT PRESENT] [FINGERS]
# | | | - LeftHandRing3 (LeftHandRing2)  --[BEAT PRESENT] [FINGERS]
# | | | - LeftHandRing4 (LeftHandRing3)  --[BEAT PRESENT] [FINGERS]
# | | | - LeftHandRing4_Nub (LeftHandRing4)  --[BEAT NOT PRESENT]
# | | - LeftHandMiddle1 (LeftHand)  --[BEAT PRESENT] [FINGERS]
# | | - LeftHandMiddle2 (LeftHandMiddle1)  --[BEAT PRESENT] [FINGERS]
# | | - LeftHandMiddle3 (LeftHandMiddle2)  --[BEAT PRESENT] [FINGERS]
# | | - LeftHandMiddle4 (LeftHandMiddle3)  --[BEAT PRESENT] [FINGERS]
# | | - LeftHandMiddle4_Nub (LeftHandMiddle4)  --[BEAT NOT PRESENT]
# | - RightShoulder (Spine3)  --[BEAT PRESENT]
# | - RightArm (RightShoulder)  --[BEAT PRESENT]
# | - RightForeArm (RightArm)  --[BEAT PRESENT]
# | - RightHand (RightForeArm)  --[BEAT PRESENT]
# | | | - RightHandIndex (RightHand)  --[BEAT PRESENT] [FINGERS]
# | | | | - RightHandThumb1 (RightHandIndex)  --[BEAT PRESENT] [FINGERS]
# | | | | - RightHandThumb2 (RightHandThumb1)  --[BEAT PRESENT] [FINGERS]
# | | | | - RightHandThumb3 (RightHandThumb2)  --[BEAT PRESENT] [FINGERS]
# | | | | - RightHandThumb4 (RightHandThumb3)  --[BEAT PRESENT] [FINGERS]
# | | | | - RightHandThumb4_Nub (RightHandThumb4)  --[BEAT NOT PRESENT]
# | | | - RightHandIndex1 (RightHandIndex)  --[BEAT PRESENT] [FINGERS]
# | | | - RightHandIndex2 (RightHandIndex1)  --[BEAT PRESENT] [FINGERS]
# | | | - RightHandIndex3 (RightHandIndex2)  --[BEAT PRESENT] [FINGERS]
# | | | - RightHandIndex4 (RightHandIndex3)  --[BEAT PRESENT] [FINGERS]
# | | | - RightHandIndex4_Nub (RightHandIndex4)  --[BEAT NOT PRESENT]
# | | - RightHandRing (RightHand)  --[BEAT PRESENT] [FINGERS]
# | | | - RightHandPinky (RightHandRing)  --[BEAT PRESENT] [FINGERS]
# | | | - RightHandPinky1 (RightHandPinky)  --[BEAT PRESENT] [FINGERS]
# | | | - RightHandPinky2 (RightHandPinky1)  --[BEAT PRESENT] [FINGERS]
# | | | - RightHandPinky3 (RightHandPinky2)  --[BEAT PRESENT] [FINGERS]
# | | | - RightHandPinky4 (RightHandPinky3)  --[BEAT PRESENT] [FINGERS]
# | | | - RightHandPinky4_Nub (RightHandPinky4)  --[BEAT NOT PRESENT]
# | | - RightHandRing1 (RightHandRing)  --[BEAT PRESENT] [FINGERS]
# | | - RightHandRing2 (RightHandRing1)  --[BEAT PRESENT] [FINGERS]
# | | - RightHandRing3 (RightHandRing2)  --[BEAT PRESENT] [FINGERS]
# | | - RightHandRing4 (RightHandRing3)  --[BEAT PRESENT] [FINGERS]
# | | - RightHandRing4_Nub (RightHandRing4)  --[BEAT NOT PRESENT]
# | - RightHandMiddle1 (RightHand)  --[BEAT PRESENT] [FINGERS]
# | - RightHandMiddle2 (RightHandMiddle1)  --[BEAT PRESENT] [FINGERS]
# | - RightHandMiddle3 (RightHandMiddle2)  --[BEAT PRESENT] [FINGERS]
# | - RightHandMiddle4 (RightHandMiddle3)  --[BEAT PRESENT] [FINGERS]
# | - RightHandMiddle4_Nub (RightHandMiddle4)  --[BEAT NOT PRESENT]
# - Neck (Spine3)  --[BEAT PRESENT]
# - Neck1 (Neck)  --[BEAT PRESENT]
# - Head (Neck1)  --[BEAT PRESENT]
# - HeadEnd (Head)  --[BEAT PRESENT]
# - HeadEnd_Nub (HeadEnd)  --[BEAT NOT PRESENT]

## 88 total: 75 BEAT, 13 extra

pymo_extras = [    # J: 13
        'HeadEnd_Nub',
        'RightHandMiddle4_Nub',
        'RightHandRing4_Nub',
        'RightHandPinky4_Nub',
        'RightHandIndex4_Nub',
        'RightHandThumb4_Nub',
        'LeftHandMiddle4_Nub',
        'LeftHandRing4_Nub',
        'LeftHandPinky4_Nub',
        'LeftHandIndex4_Nub',
        'LeftHandThumb4_Nub',
        'RightToeBaseEnd_Nub',
        'LeftToeBaseEnd_Nub']

pymo_dB_v0 = [ # J: 20, 'Hips' as root included by default
        'Spine', 
        'Spine1', 
        'Neck', 
        'Head', 
        'RightShoulder', 
        'RightArm', 
        'RightForeArm', 
        'RightHand', 
        'LeftShoulder', 
        'LeftArm', 
        'LeftForeArm', 
        'LeftHand', 
        'RightUpLeg', 
        'RightLeg', 
        'RightFoot', 
        'RightToeBase', 
        'LeftUpLeg', 
        'LeftLeg', 
        'LeftFoot', 
        'LeftToeBase'] 

pymo_dB_v0_fing = [ # J: 64, 'Hips' as root included by default
        'LeftUpLeg',
        'LeftLeg',
        'LeftFoot',
        'LeftToeBase',
        'RightUpLeg', 
        'RightLeg', 
        'RightFoot', 
        'RightToeBase',
        'Spine', 
        'Spine1', 
        'LeftShoulder', 
        'LeftArm', 
        'LeftForeArm', 
        'LeftHand', 
        'LeftHandIndex',
        'LeftHandThumb1',
        'LeftHandThumb2',
        'LeftHandThumb3',
        'LeftHandThumb4',
        'LeftHandIndex1',
        'LeftHandIndex2',
        'LeftHandIndex3',
        'LeftHandIndex4',
        'LeftHandRing',
        'LeftHandPinky1',
        'LeftHandPinky2',
        'LeftHandPinky3',  
        'LeftHandPinky4',
        'LeftHandRing1',
        'LeftHandRing2',
        'LeftHandRing3',
        'LeftHandRing4',
        'LeftHandMiddle1',
        'LeftHandMiddle2',
        'LeftHandMiddle3',
        'LeftHandMiddle4',
        'RightShoulder', 
        'RightArm', 
        'RightForeArm', 
        'RightHand',
        'RightHandIndex',
        'RightHandThumb1',
        'RightHandThumb2',
        'RightHandThumb3',
        'RightHandThumb4',
        'RightHandIndex1',
        'RightHandIndex2',
        'RightHandIndex3',
        'RightHandIndex4',
        'RightHandRing',
        'RightHandPinky1',
        'RightHandPinky2',
        'RightHandPinky3',
        'RightHandPinky4',
        'RightHandRing1',
        'RightHandRing2',
        'RightHandRing3',
        'RightHandRing4',
        'RightHandMiddle1',
        'RightHandMiddle2',
        'RightHandMiddle3',
        'RightHandMiddle4',
        'Neck',
        'Head'] 

pymo_dB_v0_fing_combined_feats =  {}  # TODO

pymo_dB_v0_combined_feats = { # F: 69
        'LeftToeBase_alpha': 1,   
        'LeftToeBase_beta': 2,  
        'LeftToeBase_gamma': 3,
        'LeftFoot_alpha': 4,  
        'LeftFoot_beta': 5,         
        'LeftFoot_gamma': 6,        
        'LeftLeg_alpha': 7,       
        'LeftLeg_beta': 8,     
        'LeftLeg_gamma': 9,        
        'LeftUpLeg_alpha': 10,   
        'LeftUpLeg_beta': 11,  
        'LeftUpLeg_gamma': 12,    
        'RightToeBase_alpha': 13,
        'RightToeBase_beta': 14, 
        'RightToeBase_gamma': 15,
        'RightFoot_alpha': 16,
        'RightFoot_beta': 17,   
        'RightFoot_gamma': 18,  
        'RightLeg_alpha': 19,  
        'RightLeg_beta': 20,   
        'RightLeg_gamma': 21,    
        'RightUpLeg_alpha': 22,
        'RightUpLeg_beta': 23,
        'RightUpLeg_gamma': 24,  
        'LeftHand_alpha': 25,  
        'LeftHand_beta': 26,    
        'LeftHand_gamma': 27,    
        'LeftForeArm_alpha': 28,     
        'LeftForeArm_beta': 29,
        'LeftForeArm_gamma': 30,  
        'LeftArm_alpha': 31,
        'LeftArm_beta': 32,   
        'LeftArm_gamma': 33,      
        'LeftShoulder_alpha': 34,    
        'LeftShoulder_beta': 35, 
        'LeftShoulder_gamma': 36, 
        'RightHand_alpha': 37,
        'RightHand_beta': 38,   
        'RightHand_gamma': 39,    
        'RightForeArm_alpha': 40,   
        'RightForeArm_beta': 41,
        'RightForeArm_gamma': 42, 
        'RightArm_alpha': 43,
        'RightArm_beta': 44,     
        'RightArm_gamma': 45,      
        'RightShoulder_alpha': 46, 
        'RightShoulder_beta': 47,
        'RightShoulder_gamma': 48, 
        'Head_alpha': 49,
        'Head_beta': 50,          
        'Head_gamma': 51,         
        'Neck_alpha': 52,        
        'Neck_beta': 53,      
        'Neck_gamma': 54,          
        'Spine1_alpha': 55,        
        'Spine1_beta': 56,      
        'Spine1_gamma': 57,      
        'Spine_alpha': 58,       
        'Spine_beta': 59,       
        'Spine_gamma': 60,        
        'Hips_alpha': 61,     
        'Hips_beta': 62,        
        'Hips_gamma': 63,         
        'Hips_Xposition': 64,       
        'Hips_Yposition': 65,     
        'Hips_Zposition': 66,     
        'Hips_dXposition': 67,    
        'Hips_dZposition': 68,   
        'Hips_dYrotation': 69}

pymo_dB_v0_emo_feats = { # F: 45
        'LeftToeBase_alpha': 1,   
        'LeftToeBase_beta': 2,  
        'LeftToeBase_gamma': 3,
        'LeftFoot_alpha': 4,  
        'LeftFoot_beta': 5,         
        'LeftFoot_gamma': 6,        
        'LeftLeg_alpha': 7,       
        'LeftLeg_beta': 8,     
        'LeftLeg_gamma': 9,        
        'LeftUpLeg_alpha': 10,   
        'LeftUpLeg_beta': 11,  
        'LeftUpLeg_gamma': 12,    
        'RightToeBase_alpha': 13,
        'RightToeBase_beta': 14, 
        'RightToeBase_gamma': 15,
        'RightFoot_alpha': 16,
        'RightFoot_beta': 17,   
        'RightFoot_gamma': 18,  
        'RightLeg_alpha': 19,  
        'RightLeg_beta': 20,   
        'RightLeg_gamma': 21,    
        'RightUpLeg_alpha': 22,
        'RightUpLeg_beta': 23,
        'RightUpLeg_gamma': 24,
        'Head_alpha': 49,
        'Head_beta': 50,          
        'Head_gamma': 51,         
        'Neck_alpha': 52,        
        'Neck_beta': 53,      
        'Neck_gamma': 54,          
        'Spine1_alpha': 55,        
        'Spine1_beta': 56,      
        'Spine1_gamma': 57,      
        'Spine_alpha': 58,       
        'Spine_beta': 59,       
        'Spine_gamma': 60,        
        'Hips_alpha': 61,     
        'Hips_beta': 62,        
        'Hips_gamma': 63,         
        'Hips_Xposition': 64,       
        'Hips_Yposition': 65,     
        'Hips_Zposition': 66,     
        'Hips_dXposition': 67,    
        'Hips_dZposition': 68,   
        'Hips_dYrotation': 69}

pymo_dB_v0_con_feats = { # F: 24
        'LeftHand_alpha': 25,  
        'LeftHand_beta': 26,    
        'LeftHand_gamma': 27,    
        'LeftForeArm_alpha': 28,     
        'LeftForeArm_beta': 29,
        'LeftForeArm_gamma': 30,  
        'LeftArm_alpha': 31,
        'LeftArm_beta': 32,   
        'LeftArm_gamma': 33,      
        'LeftShoulder_alpha': 34,    
        'LeftShoulder_beta': 35, 
        'LeftShoulder_gamma': 36, 
        'RightHand_alpha': 37,
        'RightHand_beta': 38,   
        'RightHand_gamma': 39,    
        'RightForeArm_alpha': 40,   
        'RightForeArm_beta': 41,
        'RightForeArm_gamma': 42, 
        'RightArm_alpha': 43,
        'RightArm_beta': 44,     
        'RightArm_gamma': 45,      
        'RightShoulder_alpha': 46, 
        'RightShoulder_beta': 47,
        'RightShoulder_gamma': 48}

j_list_dict = { "joint_name_list_225": joint_name_list_225,
                "joint_name_list_186": joint_name_list_186, 
                "joint_name_list_27": joint_name_list_27, 
                "joint_name_list_27_v3": joint_name_list_27_v3, 
                "spine_neck_141": spine_neck_141, 
                "remaining_j_v0": remaining_j_v0,
                "audio_sync_j_v0": audio_sync_j_v0,
                "combined_v0": combined_v0,
                "pymo_extras": pymo_extras,
                "pymo_dB_v0": pymo_dB_v0,
                "pymo_dB_v0_con_feats": pymo_dB_v0_con_feats,
                "pymo_dB_v0_emo_feats": pymo_dB_v0_emo_feats,
                "pymo_dB_v0_combined_feats": pymo_dB_v0_combined_feats,
                "pymo_dB_v0_fing": pymo_dB_v0_fing,
                "pymo_dB_v0_fing_combined_feats": pymo_dB_v0_fing_combined_feats,
                "smpl_skeleton": smpl_skeleton,
                "smplh_skeleton": smplh_skeleton,
                "smplx_skeleton": smplx_skeleton,
                "supr_skeleton": supr_skeleton,
                "KINECT_V2": KINECT_V2,
                "JOINT_MAP": JOINT_MAP,
                "BP2KA": BP2KA
                }

def list_check(j_list):
    count_joints = 0
    count_rotations = 0
    for aa, j_numbers in j_list.items():
        count_joints += 1
        count_rotations += j_numbers
    print('[PREPROCESS] joints:', count_joints, 'freedom:', count_rotations)
    return count_joints, count_rotations

def get_mean_pose(bvh_save_path, verbose=False):
    bvh_files_dirs = sorted(glob.glob(f'{bvh_save_path}/*.bvh'))
    data_all, data_all_c, data_all_e, data_all_com = [], [], [], []
    div_flag = True
    for i, bvh_files_dir in tqdm(enumerate(bvh_files_dirs), desc='[PREPROCESS] Pose mean, std compute...', total=len(bvh_files_dirs)):
        with open(bvh_files_dir, 'r') as pose_data:
            for j, line in enumerate(pose_data.readlines()):
                data = np.fromstring(line, dtype=float, sep=' ') 
                # print(i, '/', len(bvh_files_dirs), ' ', j-1) if verbose else None
                filetype = bvh_files_dir.split('/')[-1].split('.')[0].split('_')[-1]
                if filetype == "con":
                    data_all_c.append(data)
                elif filetype == "emo":
                    data_all_e.append(data)
                elif filetype == "combined":
                    data_all_com.append(data)
                else:
                    div_flag = False
                    data_all.append(data)
                    
    if div_flag:
        data_all_c = np.array(data_all_c)
        m_c = np.mean(data_all_c, axis=0)
        s_c = np.std(data_all_c, axis=0)
        print("data_all_c", data_all_c.shape, "m_c", m_c.shape, "s_c", s_c.shape) if verbose else None
        with open(f'{bvh_save_path}/mean_con.npy', 'wb') as f:
            np.save(f, m_c)
        with open(f'{bvh_save_path}/std_con.npy', 'wb') as f:
            np.save(f, s_c)
        _get_mean_pose_helper(s_c) if verbose else None
        
        data_all_e = np.array(data_all_e)
        m_e = np.mean(data_all_e, axis=0)
        s_e = np.std(data_all_e, axis=0)
        print("data_all_e", data_all_e.shape, "m_e", m_e.shape, "s_e", s_e.shape) if verbose else None
        with open(f'{bvh_save_path}/mean_emo.npy', 'wb') as f:
            np.save(f, m_e)
        with open(f'{bvh_save_path}/std_emo.npy', 'wb') as f:
            np.save(f, s_e)
        _get_mean_pose_helper(s_e) if verbose else None
        
        data_all_com = np.array(data_all_com)
        m_com = np.mean(data_all_com, axis=0)
        s_com = np.std(data_all_com, axis=0)
        print("data_all_com", data_all_com.shape, "m_com", m_com.shape, "s_com", s_com.shape) if verbose else None
        with open(f'{bvh_save_path}/mean_combined.npy', 'wb') as f:
            np.save(f, m_com)
        with open(f'{bvh_save_path}/std_combined.npy', 'wb') as f:
            np.save(f, s_com)
        _get_mean_pose_helper(s_com) if verbose else None
        
    else:
        data_all = np.array(data_all)
        m = np.mean(data_all, axis=0)
        s = np.std(data_all, axis=0)
        print("data_all", data_all.shape, "m", m.shape, "s", s.shape) if verbose else None
        with open(f'{bvh_save_path}/mean.npy', 'wb') as f:
            np.save(f, m)
        with open(f'{bvh_save_path}/std.npy', 'wb') as f:
            np.save(f, s)
        _get_mean_pose_helper(s) if verbose else None

    print("Mean, std extracted for given bvh types.") if verbose else None

def _get_mean_pose_helper(std_pose):
    """
    TODO: check this function
    """
    count = 0
    for i in range(len(std_pose)//3):
        count += 1
        max_val = max(std_pose[i*3:i*3+2]) 
        min_val = min(std_pose[i*3:i*3+2])
        print(max_val, min_val)
        if min_val < 0:
            print(count, min_val)

def transfer2target(bvh_file, bvh_save_path, config):
    
    # with open(bvh_file) as f:
    #     mocap = Bvh(f.read())        
        # print([str(item) for item in mocap.root]) #  mocap tree ['HIERARCHY', 'ROOT Hips', 'MOTION', 'Frames: 9000', 'Frame Time: 0.008333']
        # print(next(mocap.root.filter('ROOT'))['OFFSET']) # root offset
        # print(mocap.joint_offset('LeftUpLeg'))
        # print(mocap.nframes)
        # print(mocap.frame_time)
        # print(mocap.joint_channels('Hips')) # ['Xposition', 'Yposition', 'Zposition', 'Xrotation', 'Yrotation', 'Zrotation']
        # print(mocap.joint_channels('LeftUpLeg')) # ['Xrotation', 'Yrotation', 'Zrotation']
        # print(mocap.frame_joint_channel(22, 'LeftHandThumb1', 'Xrotation'))
        # print(mocap.get_joints_names())
        # print(mocap.joint_parent_index('RightUpLeg'))
        # print(mocap.joint_parent('RightUpLeg').name)
        # print(mocap.joint_direct_children('LeftShoulder'))

    bvh_flag = config["DATA_PARAM"]["Bvh"]["con_emo_div"]["use"]
    
    if bvh_flag:
        j_list_con = j_list_dict[config["DATA_PARAM"]["Bvh"]["con_emo_div"]["con"]] 
        j_list_emo = j_list_dict[config["DATA_PARAM"]["Bvh"]["con_emo_div"]["emo"]]
        j_list_combined = j_list_dict[config["DATA_PARAM"]["Bvh"]["con_emo_div"]["combined"]]
    else:
        j_list = j_list_dict[config["DATA_PARAM"]["Bvh"]["joint_name_list"]]
    
    # list_check(j_list) # joints: 9 freedom: 27 
    
    with open(bvh_file, "r") as pose_data:
        d_e, d_c, d_combined, d = [], [], [], []
        for a, line in enumerate(pose_data.readlines()[430:]): # 0:428 HIERARCHY and 429:-- MOTION
            if not a:
                words = line.split()
                FPS = math.ceil(1/float(words[-1])) 
                # factor = math.ceil(FPS / config["DATA_PARAM"]["Bvh"]["fps"]) # f5 less data
                factor = math.ceil(FPS // config["DATA_PARAM"]["Bvh"]["fps"]) # f4 more data
                # raise Exception("Original FPS: ", FPS,  "Reduced factor: ", factor) # Original FPS:  121 Reduced factor:  4
            else:
                if a % factor != 1 and factor != 1:
                    continue
                data = np.fromstring(line, dtype=float, sep=' ')
                
                if bvh_flag:              
                    data_emo = _transfer_helper(j_list_emo, data)
                    d_e.append(data_emo)
                    
                    data_con = _transfer_helper(j_list_con, data)
                    d_c.append(data_con)
                    
                    data_combined = _transfer_helper(j_list_combined, data)
                    d_combined.append(data_combined)
                else:
                    data_nondiv = _transfer_helper(j_list, data)
                    d.append(data_nondiv)

    if bvh_flag:
        _transfer_saver(bvh_save_path, bvh_file, data_each_file=d_c, tag="con")
        _transfer_saver(bvh_save_path, bvh_file, data_each_file=d_e, tag="emo")
        _transfer_saver(bvh_save_path, bvh_file, data_each_file=d_combined, tag="combined")
    else:
        _transfer_saver(bvh_save_path, bvh_file, data_each_file=d, tag="")
        
    get_mean_pose(bvh_save_path)
                    
def _transfer_saver(bvh_save_path, bvh_file, data_each_file, tag=None):
    file_path_base = bvh_save_path / bvh_file.split("/")[-1].split(".")[0] 
    file_path = str(file_path_base) + "_" + tag + ".bvh" if tag else str(file_path_base) + ".bvh"
    with open(file_path, "w+") as write_file:
        for line_data in data_each_file:
            line_data = np.array2string(line_data, max_line_width=np.inf, precision=6, separator=' ', suppress_small=False)
            write_file.write(line_data[1:-2]+'\n')

def _transfer_helper(j_list, data, data_rotation=np.zeros((1))):
    for k, v in j_list.items():
        data_rotation = np.concatenate((data_rotation, data[beat_joints[k][1] - v:beat_joints[k][1]]))
    return data_rotation[1:]

def get_mean_std(processed_path, target_list, bvh_fps, dwnsmpl_fac, disentangled, verbose=False):
    mean_std = {}
    processed_bvhs = Path(processed_path) / f"bvh-fps{bvh_fps}-{target_list[0]}-div-{dwnsmpl_fac}" 
    if verbose: print(f"[DIFF VIZ] Loading mean and std from {processed_bvhs}")   
    if disentangled:
        mean_std["mean_con"] = np.load(processed_bvhs / "mean_con.npy")
        mean_std["std_con"] = np.load(processed_bvhs / "std_con.npy")
        mean_std["mean_emo"] = np.load(processed_bvhs / "mean_emo.npy")
        mean_std["std_emo"] = np.load(processed_bvhs / "std_emo.npy")
    else:
        mean_std["mean_combined"] = np.load(processed_bvhs / "mean_combined.npy")
        mean_std["std_combined"] = np.load(processed_bvhs / "std_combined.npy")
    return mean_std
    
def normalize(m, mean, std, verbose=False):
    
    frames = m.shape[0]
    joints = m.shape[1]
    normalized = m.clone()
    
    for j in range(joints):
        normalized[:,j] = (m[:,j] - mean[j]) / std[j]
        
    print("[DIFF] Normalized pose") if verbose else None
    return normalized

def unnormalize(pose, mean_std, disentangled, emo_tag, verbose=False):
    
    unnormalized_pose = pose.clone()
    if disentangled:
        if emo_tag:
            for j in range(pose.shape[1]):
                unnormalized_pose[:,j] = (pose[:,j] * mean_std["std_emo"][j]) + mean_std["mean_emo"][j]
        else:
            for j in range(pose.shape[1]):
                unnormalized_pose[:,j] = (pose[:,j] * mean_std["std_con"][j]) + mean_std["mean_con"][j]
    else:
        for j in range(pose.shape[1]):
            unnormalized_pose[:,j] = (pose[:,j] * mean_std["std_combined"][j]) + mean_std["mean_combined"][j]
        
    print("[DIFF VIZ] Unnormalized ", "dis-emo " if emo_tag else "combo " if not disentangled else "dis-con ", "pose") if verbose else None
    return unnormalized_pose
          
def check_static_motion(verbose=False):
    # TODO
    pass

def check_pose_diff(verbose=False):
    # TODO
    pass

def check_spine_angle(verbose=False):
    # TODO
    pass

def result2target_vis(source_bvh_path, ori_list, target_list, viz_path, filename, source, bvh_fps, disentangled, debug):
   
    file_content_length = 431

    src_file = source_bvh_path
    tgt_file = viz_path / f"{filename}_{source}_viz.bvh"
    raw_con_file = viz_path / f"{filename}_{source}_raw_con.bvh"
    raw_emo_file = viz_path / f"{filename}_{source}_raw_emo.bvh"
    raw_combo_file = viz_path / f"{filename}_{source}_raw_combo.bvh"
    
    # copy source skeleton hierarchy, offset motion frame
    with open(str(src_file), "r") as src_data:
        src_data_file = src_data.readlines()
        offset_motion_frame = np.fromstring(src_data_file[file_content_length], dtype=float, sep=' ')

    if disentangled:
        
        # con predictions
        with open(str(raw_con_file), "r") as con_data:
            con_data_file = con_data.readlines()
            con_frames = len(con_data_file)
        
        # emo predictions
        with open(str(raw_emo_file), "r") as emo_data:
            emo_data_file = emo_data.readlines()
            emo_frames = len(emo_data_file)
        
        # frame count
        assert con_frames == emo_frames, "[DIFF VIZ/BVH Util] Frame count mismatch between con and emo predictions"
        frame_count = con_frames
        frame_time = 1 / bvh_fps
 
        # data rotations for con and emo predictions
        con_tgt = j_list_dict[target_list[0]]
        emo_tgt = j_list_dict[target_list[1]]
        data_each_file = []
        data_rotation = offset_motion_frame.copy()
        for c, con_line in enumerate(con_data_file):
            if not c: pass
            else:
                con_pred = np.fromstring(con_line, dtype=float, sep=' ')
                emo_pred = np.fromstring(emo_data_file[c], dtype=float, sep=' ')
                for iii, (k, v) in enumerate(con_tgt.items()):
                    data_rotation[ori_list[k][1] - v:ori_list[k][1]] = con_pred[iii*3:iii*3+3]
                for iii, (k, v) in enumerate(emo_tgt.items()):
                    data_rotation[ori_list[k][1] - v:ori_list[k][1]] = emo_pred[iii*3:iii*3+3]
                data_each_file.append(data_rotation)
  
    else: 
        
        # combo predictions
        with open(str(raw_combo_file), "r") as combo_data:
            combo_data_file = combo_data.readlines()
            combo_frames = len(combo_data_file)
        
        # frame count
        frame_count = combo_frames
        frame_time = 1 / bvh_fps
 
        # data rotations for con and emo predictions
        combo_tgt = j_list_dict[target_list[2]]
        data_each_file = []
        data_rotation = offset_motion_frame.copy()
        for c, combo_line in enumerate(combo_data_file):
            if not c: pass
            else: 
                combo_pred = np.fromstring(combo_line, dtype=float, sep=' ')
                for iii, (k, v) in enumerate(combo_tgt.items()):
                    data_rotation[ori_list[k][1] - v:ori_list[k][1]] = combo_pred[iii*3:iii*3+3]
                data_each_file.append(data_rotation)
    
    # paste skeleton hierarchy, frame count, frame time
    src_data_file = src_data_file[:file_content_length]
    src_data_file[file_content_length-2] = 'Frames: ' + str(frame_count) + '\n'
    src_data_file[file_content_length-1] = 'Frame Time: ' + str(frame_time) + '\n'
    
    # write everthing to target bvh file
    tgt_write_file = open(str(tgt_file), "w+")
    tgt_write_file.writelines(i for i in src_data_file)
    for line_data in data_each_file:
        line_data = np.array2string(line_data, max_line_width=np.inf, precision=6, suppress_small=False, separator=' ')
        tgt_write_file.write(line_data[1:-1] + '\n')
    tgt_write_file.close()
    if debug: print("[DIFF VIZ] Disentangled result written to ", tgt_file)
        
    return tgt_file

############################### PyMO Utils #####################################

def pymo_pipeline(all_data, bvh_save_path, val_held_out, test_held_out, config, verbose=False, pymo_verbose=False):                                                        
    if verbose: print("[PyMO BVH] Processing BVH files...")
    if type(all_data) == dict:
        all_bvh_files = []
        for i in all_data.keys():
            for j in all_data[i].keys():
                all_bvh_files.append(all_data[i][j]["bvh"][0])
    else: assert type(all_data) == list; all_bvh_files = all_data
    # pymo_parsed_errors = ["0_16_16", "0_82_82", "0_96_96", "0_88_88"]                 # fixed
    # all_bvh_files = [ x for x in all_bvh_files if "_".join(x.split("/")[-1].split(".")[0].split("_")[2:]) not in pymo_parsed_errors] # fixed
    if not len(glob.glob(str(bvh_save_path) + "/*")) == len(all_bvh_files) + 1: # +1 for data_pipe.sav 
        _pymo_extract_joint_angles(all_bvh_files, bvh_save_path, config, verbose, pymo_verbose)
    else: 
        print("[PyMO BVH] BVH files already processed, skipping...") if verbose else None
    all_npzs = [f for f in bvh_save_path.glob("*.npz")]
    train_npz = [f for f in all_npzs if f.name.split("_")[1] not in val_held_out and f.name.split("_")[1] not in test_held_out]
    val_npz = [f for f in all_npzs if f.name.split("_")[1] in val_held_out]
    test_npz = [f for f in all_npzs if f.name.split("_")[1] in test_held_out]                                                             
    # train_motion = _pymo_import_and_slice(bvh_save_path, config, verbose)          # Slicing done in dm.py
    train_motion, _, train_npz_paths, train_npz_lengths = _pymo_import_data(train_npz, slicing=False, verbose=verbose)                                                                                      
    train_motion, output_scaler = _pymo_fit_and_standardize(train_motion, verbose)                                                                   
    val_motion, _, val_npz_paths, val_npz_lengths = _pymo_import_data(val_npz, slicing=False, verbose=verbose)                                                                          
    val_motion = _pymo_standardize(val_motion, output_scaler, verbose)                                                                                
    test_motion, _, test_npz_paths, test_npz_lengths = _pymo_import_data(test_npz, slicing=False, verbose=verbose)                                                                  
    test_motion = _pymo_standardize(test_motion, output_scaler, verbose)
    jl.dump(output_scaler, str(bvh_save_path / "output_scaler.sav"))
    np.savez(str(bvh_save_path / "train_motion.npz"), clips=train_motion)                                                                             
    np.savez(str(bvh_save_path / "val_motion.npz"), clips=val_motion)                                                                                          
    np.savez(str(bvh_save_path / "test_motion.npz"), clips=test_motion)    
    with open(str(bvh_save_path / "train_npz_paths.pkl"), "wb") as f: pickle.dump(train_npz_paths, f)
    with open(str(bvh_save_path / "val_npz_paths.pkl"), "wb") as f: pickle.dump(val_npz_paths, f)
    with open(str(bvh_save_path / "test_npz_paths.pkl"), "wb") as f: pickle.dump(test_npz_paths, f)
    with open(str(bvh_save_path / "train_npz_lengths.pkl"), "wb") as f: pickle.dump(train_npz_lengths, f)
    with open(str(bvh_save_path / "val_npz_lengths.pkl"), "wb") as f: pickle.dump(val_npz_lengths, f)
    with open(str(bvh_save_path / "test_npz_lengths.pkl"), "wb") as f: pickle.dump(test_npz_lengths, f)                                                                                   
    process_bvh_flag = False
    return process_bvh_flag

def pymo_inverse_pipeline(motion_data, bvh_save_path, viz_path, actor_take, bvh_fps, verbose, viz_only=False):
    # raise Exception(motion_data.shape) # Exception: (1620, 69)
    output_scaler = jl.load(str(bvh_save_path / "output_scaler.sav"))    
    anim_clips = _pymo_inv_standardize(np.expand_dims(motion_data, axis=0), output_scaler, verbose) 
    if not viz_only: np.savez(str(viz_path / (actor_take + ".npz")), clips=anim_clips) 
    bvh_file = _pymo_write_bvh(anim_clips, actor_take, bvh_save_path, viz_path, bvh_fps, verbose)
    return bvh_file

def pymo_feats2joints(motion_data, bvh_save_path, d):
    output_scaler_sk = jl.load(str(bvh_save_path / "output_scaler.sav")) 
    output_scaler_torch = sk2torch.wrap(output_scaler_sk).to(d)
    anim_clips = _pymo_inv_tensor_standardize(motion_data, output_scaler_torch, verbose=False)
    # data_pipe_sk = jl.load(str(bvh_save_path / "data_pipe.sav"))              # unsupported sklearn estimator type: PyMO
    # data_pipe_torch = sk2torch.wrap(data_pipe_sk).to(d)                       # unsupported sklearn estimator type: PyMO
    # inv_data = data_pipe_torch.inverse_transform(anim_clips)                  # unsupported sklearn estimator type: PyMO
    return anim_clips

def pymo_con_emo_split(motion1, motion2=None, split=False, cfg=None, verbose=False):
    con_tgt = j_list_dict[cfg["DATA_PARAM"]["Bvh"]["pymo_based"]["con"][0]] # 24  
    emo_tgt = j_list_dict[cfg["DATA_PARAM"]["Bvh"]["pymo_based"]["emo"][0]] # 45
    combined_tgt = j_list_dict[cfg["DATA_PARAM"]["Bvh"]["pymo_based"]["combined"][0]] # 69
    if split:
        frames, feats = motion1.shape
        assert feats == len(combined_tgt), "[PyMO BVH] Joint list mismatch for combined joint list"
        con_motion, emo_motion = [], []
        for k, v in combined_tgt.items():
            if k in con_tgt:
                con_motion.append(motion1[:, v-1])
            elif k in emo_tgt:
                emo_motion.append(motion1[:, v-1])
            else:
                raise Exception("[PyMO BVH] Joint list mismatch")
        con_motion = np.stack(con_motion, axis=1)
        emo_motion = np.stack(emo_motion, axis=1)
        combined_motion = motion1
        assert con_motion.shape[1] == len(con_tgt) and emo_motion.shape[1] == len(emo_tgt), "[PyMO BVH] Joint list mismatch for split joint lists"
        assert frames == con_motion.shape[0] == emo_motion.shape[0], "[PyMO BVH] Frame mismatch for split motion data"
        if verbose: print("[PyMO BVH] Splitting combined motion data into con and emo...")
    else:
        assert None not in [motion1, motion2], "[PyMO BVH] Motion data not provided for emo-con split"
        req_grad = motion1.requires_grad
        if req_grad:                                                            # https://discuss.pytorch.org/t/get-values-from-tensor-without-detaching/138465
            motion1 = motion1.clone().detach().cpu().numpy()
            motion2 = motion2.clone().detach().cpu().numpy()
        frames1, feats1 = motion1.shape # con
        frames2, feats2 = motion2.shape # emo
        assert feats1 == len(con_tgt) and feats2 == len(emo_tgt), "[PyMO BVH] Joint list mismatch for emo-con split, args: Motion1=con, Motion2=emo"
        assert len(combined_tgt) == feats1 + feats2, "[PyMO BVH] Joint list mismatch for emo-con split"
        combined_motion = []
        con_idx = list(con_tgt.values())
        emo_idx = list(emo_tgt.values())
        for i in range(len(combined_tgt)):
            if i+1 in con_idx:
                combined_motion.append(motion1[:, con_idx.index(i+1)])
            elif i+1 in emo_idx:
                combined_motion.append(motion2[:, emo_idx.index(i+1)])
            else:
                raise Exception("[PyMO BVH] Joint list mismatch")
        combined_motion = np.stack(combined_motion, axis=1)
        con_motion, emo_motion = motion1, motion2
        assert frames1 == frames2 == combined_motion.shape[0], "[PyMO BVH] Frame mismatch for combined motion data"
        assert combined_motion.shape[1] == len(combined_tgt), "[PyMO BVH] Joint list mismatch for combined joint list"
        if verbose: print("[PyMO BVH] Combining con and emo motion data...")
    return con_motion, emo_motion, combined_motion       

def _pymo_standardize(motion_data, output_scaler, verbose):                                                                                                   
    if verbose: print("[PyMO BVH] Standardizing val and test motion data...")
    shape = motion_data.shape
    flat = motion_data.copy().reshape((shape[0]*shape[1], shape[2]))
    scaled = output_scaler.transform(flat).reshape(shape)
    return scaled

def _pymo_write_bvh(anim_clips, out_file_name, bvh_save_path, viz_path, bvh_fps, verbose):
    if verbose: print("[PyMO BVH] Writing BVH files...")
    data_pipe = jl.load(str(bvh_save_path / "data_pipe.sav"))
    inv_data = data_pipe.inverse_transform(anim_clips)
    writer = BVHWriter()
    bvh_file = str(viz_path / (out_file_name + ".bvh"))
    with open(bvh_file, 'w') as f:
        writer.write(inv_data[0], f, framerate=bvh_fps)
    return bvh_file

def _pymo_inv_standardize(data, scaler, verbose=False):
    shape = data.shape
    flat = data.copy().reshape((shape[0]*shape[1], shape[2]))
    scaled = scaler.inverse_transform(flat).reshape(shape)
    if verbose: print("[PyMO BVH] Inverse standardizing data...")
    return scaled    

def _pymo_inv_tensor_standardize(data, scaler, verbose=False):
    shape = data.shape
    flat = rearrange(data, 'b f d -> (b f) d')
    scaled = scaler.inverse_transform(flat).reshape(shape)
    if verbose: print("[PyMO BVH] Inverse standardizing data...")
    return scaled    

def _pymo_fit_and_standardize(data, verbose=False):
    if verbose: print("[PyMO BVH] Standardizing data...")
    shape = data.shape
    flat = data.copy().reshape((shape[0]*shape[1], shape[2]))
    scaler = StandardScaler().fit(flat)
    scaled = scaler.transform(flat).reshape(shape)
    return scaled, scaler    

def _pymo_import_and_slice(bvh_save_path, config, verbose=False):               # To be tested
    for i, bvh_file in enumerate(bvh_save_path.glob("*.npz")):
        data, n_motion_feats, _, _ = _pymo_import_data(bvh_file, verbose=verbose)
        sliced = _pymo_slice_data(data, config, verbose)
        if i==0:
            out_data = sliced
        else:
            out_data = np.concatenate((out_data, sliced), axis=0)
    return out_data # out_data[:,:,:n_motion_feats]

def _pymo_import_data(bvh_file, slicing=True, verbose=False):
    if slicing:
        if verbose: print("[PyMO BVH] Importing data from ", bvh_file, " next slicing...")
        motion_data = np.load(str(bvh_file))["clips"].astype(np.float32)
        n_motion_feats = motion_data.shape[1]
    else:
        motion_data = []
        npz_paths = [f for f in bvh_file if f.suffix == ".npz"]
        for i, bvh_file in enumerate(npz_paths):
            if verbose: print("[PyMO BVH] Importing data from ", str(bvh_file).split("/")[-1], "...")
            motion_data.append(np.load(str(bvh_file), allow_pickle=True)["clips"].astype(np.float32))
    max_time_len = np.max([m.shape[0] for m in motion_data])
    motion_len_per_npz = [m.shape[0] for m in motion_data]
    n_samples = len(motion_data)
    n_motion_feats = motion_data[0].shape[1]
    # for all n_samples, pad n_motion_feats to max_time_len by 0.0
    for i in range(n_samples):
        motion_data[i] = np.pad(motion_data[i], ((0, max_time_len-motion_data[i].shape[0]), (0, 0)), 'constant', constant_values=0.0)
    motion_data = np.array(motion_data)
    return motion_data, n_motion_feats, npz_paths, motion_len_per_npz

def _pymo_slice_data(data, config, verbose=False):    
    overlap = config["DATA_PARAM"]["Bvh"]["overlap"] # FIXME: 120
    window_size = config["DATA_PARAM"]["Bvh"]["window_size"] # FIXME: 0.5
    nframes = data.shape[0]
    overlap_frames = (int)(overlap*window_size)
    n_sequences = (nframes-overlap_frames)//(window_size-overlap_frames)
    sliced = np.zeros((n_sequences, window_size, data.shape[1])).astype(np.float32)
    if n_sequences>0:
        for i in range(0,n_sequences):
            frameIdx = (window_size-overlap_frames) * i
            sliced[i,:,:] = data[frameIdx:frameIdx+window_size,:].copy()
    else:
        raise ValueError("[PyMO BVH] Not enough frames to slice")
    if verbose: print("[PyMO BVH] Sliced data into ", sliced.shape[0], " sequences of length ", sliced.shape[1])
    return sliced

def _pymo_extract_joint_angles(all_bvh_files, bvh_save_path, config, verbose=False, pymo_verbose=False):
    p = BVHParser()
    data_all = list()
    pbar = tqdm(all_bvh_files, leave=False)
    for f in pbar:
        pbar.set_description("[PyMO BVH] Extract eulers-part from %s" % f.split("/")[-1].split(".")[0])
        data_all.append(p.parse(f))
    if verbose: print("[PyMO BVH] Processing using sklearn pipeline...")
    joints_version = config["DATA_PARAM"]["Bvh"]["pymo_based"]["version"]
    if "_fing" in joints_version: selected_joints = j_list_dict["pymo_dB_v0_fing"]
    else: selected_joints = j_list_dict["pymo_dB_v0"]
    if "_v0_" in joints_version:
        data_pipe = Pipeline([
            ('dwnsampl', DownSampler(tgt_fps=config["DATA_PARAM"]["Bvh"]["fps"],  keep_all=False, pymo_verbose=pymo_verbose)),
            # ('mir', Mirror(axis='X', append=True)),                               # Not used
            ('jtsel', JointSelector(selected_joints, include_root=True, pymo_verbose=pymo_verbose)),
            ('root', RootTransformer('pos_rot_deltas', position_smoothing=5, rotation_smoothing=10, pymo_verbose=pymo_verbose)), # abdolute_translation_deltas (full body), hip_centric (half body?)
            ('exp', MocapParameterizer('expmap', pymo_verbose=pymo_verbose)),       # data representation: 'euler', 'expmap'
            # ('cnst', ConstantsRemover()),                                         # Not used
            ('np', Numpyfier(pymo_verbose=pymo_verbose))
            ])
    elif "_v1_" in joints_version: # no root transformation
        data_pipe = Pipeline([
            ('dwnsampl', DownSampler(tgt_fps=config["DATA_PARAM"]["Bvh"]["fps"],  keep_all=False, pymo_verbose=pymo_verbose)),
            ('jtsel', JointSelector(selected_joints, include_root=True, pymo_verbose=pymo_verbose)),
            ('exp', MocapParameterizer('expmap', pymo_verbose=pymo_verbose)),     
            ('np', Numpyfier(pymo_verbose=pymo_verbose))
            ])
    elif "_v2_" in joints_version: # root transformation without smoothing (Used, Outcome: most realistic compared to v3-5)
        data_pipe = Pipeline([
            ('dwnsampl', DownSampler(tgt_fps=config["DATA_PARAM"]["Bvh"]["fps"],  keep_all=False, pymo_verbose=pymo_verbose)),
            ('jtsel', JointSelector(selected_joints, include_root=True, pymo_verbose=pymo_verbose)),
            ('root', RootTransformer('pos_rot_deltas', position_smoothing=0, rotation_smoothing=0, pymo_verbose=pymo_verbose)), 
            ('exp', MocapParameterizer('expmap', pymo_verbose=pymo_verbose)),      
            ('np', Numpyfier(pymo_verbose=pymo_verbose))
            ])
    elif "_v3_" in joints_version: # root transformation without smoothing, with rootTrans (Outcome: sliding slowly)
        data_pipe = Pipeline([
            ('dwnsampl', DownSampler(tgt_fps=config["DATA_PARAM"]["Bvh"]["fps"],  keep_all=False, pymo_verbose=pymo_verbose)),
            ('jtsel', JointSelector(selected_joints, include_root=True, pymo_verbose=pymo_verbose)),
            ('root', RootTransformer('pos_rot_deltas', position_smoothing=0, rotation_smoothing=0, pymo_verbose=pymo_verbose, keep_rootTrans=True)), 
            ('exp', MocapParameterizer('expmap', pymo_verbose=pymo_verbose)),      
            ('np', Numpyfier(pymo_verbose=pymo_verbose))
            ])
    elif "_v4_" in joints_version: # pos_rot_deltas_v1 without forwardY (Outcome: sliding alot, rotate mismatch with v5)
        data_pipe = Pipeline([
            ('dwnsampl', DownSampler(tgt_fps=config["DATA_PARAM"]["Bvh"]["fps"],  keep_all=False, pymo_verbose=pymo_verbose)),
            ('jtsel', JointSelector(selected_joints, include_root=True, pymo_verbose=pymo_verbose)),
            ('root', RootTransformer('pos_rot_deltas_v1', position_smoothing=0, rotation_smoothing=0, pymo_verbose=pymo_verbose, keep_rootTrans=True)), 
            ('exp', MocapParameterizer('expmap', pymo_verbose=pymo_verbose)),      
            ('np', Numpyfier(pymo_verbose=pymo_verbose))
            ])
    elif "_v5_" in joints_version: # pos_rot_deltas_v1 with forwardY (Outcome: sliding alot, rotate mismatch with v4)
        data_pipe = Pipeline([
            ('dwnsampl', DownSampler(tgt_fps=config["DATA_PARAM"]["Bvh"]["fps"],  keep_all=False, pymo_verbose=pymo_verbose)),
            ('jtsel', JointSelector(selected_joints, include_root=True, pymo_verbose=pymo_verbose)),
            ('root', RootTransformer('pos_rot_deltas_v1', position_smoothing=0, rotation_smoothing=0, pymo_verbose=pymo_verbose, keep_rootTrans=True, keep_forwardY=True)), 
            ('exp', MocapParameterizer('expmap', pymo_verbose=pymo_verbose)),      
            ('np', Numpyfier(pymo_verbose=pymo_verbose))
            ])
    out_data = data_pipe.fit_transform(data_all)
    jl.dump(data_pipe, str(bvh_save_path / 'data_pipe.sav'))    
    # for f in all_bvh_files:                                                   # Modified the loop due to index error in "engall" processing
    #     filename = f.split("/")[-1].split(".")[0]
    #     np.savez(str(bvh_save_path / f"{filename}.npz"), clips=out_data[all_bvh_files.index(f)])
    #     if verbose: print(f"[PyMO BVH] Saved {filename}.npz")
    for i, f in enumerate(all_bvh_files):
        filename = f.split("/")[-1].split(".")[0]
        np.savez(str(bvh_save_path / f"{filename}.npz"), clips=out_data[i])
        if verbose: print(f"[PyMO BVH] Saved {filename}.npz")
    if verbose: print(f"[PyMO BVH] Saved all .npz files")
    
def pymo_smpl_jt_extractor(all_bvh_files: list, config: dict, 
                           bvh_save_path: str, save_data_pipe: bool,
                           pymo_based: bool):
    """
        BVH to POS for SMPLH fitting process
    """
    
    if pymo_based:
        p = BVHParser()
        data_all = list()
        pbar = tqdm(all_bvh_files, leave=False)
        for f in pbar:
            pbar.set_description("[PyMO SMPL XTRACT] Extract eulers-part from %s" % f.split("/")[-1].split(".")[0])
            data_all.append(p.parse(f))
        selected_joints = j_list_dict[config["TRAIN_PARAM"]["motionprior"]["amass_joints"]] 
        data_pipe = Pipeline([
            ('exp', MocapParameterizer('position')),  # data representation: 'euler', 'expmap'
            # ('rcpn', RootCentricPositionNormalizer()),
            # ('root', RootTransformer('abdolute_translation_deltas', position_smoothing=5, rotation_smoothing=10, pymo_verbose=False)), # abdolute_translation_deltas (full body), hip_centric (half body?)
            ('dwnsampl', DownSampler(tgt_fps=config["DATA_PARAM"]["Bvh"]["fps"],  keep_all=False, pymo_verbose=False)),
            ('jtsel', JointSelector(selected_joints, include_root=True, pymo_verbose=False)),
            ('np', Numpyfier(pymo_verbose=False))
            ])
        out_data = data_pipe.fit_transform(data_all)
        if save_data_pipe: jl.dump(data_pipe, str(bvh_save_path / 'data_pipe.sav'))
        for i, f in enumerate(all_bvh_files):
            filename = f.split("/")[-1].split(".")[0]
            bvh_pos = out_data[i]
            arr_reshaped = bvh_pos.reshape(bvh_pos.shape[0], 22, -1)[25:50]
            np.save(str(bvh_save_path / f"{filename}_pymo.npy"), arr=arr_reshaped)
    else: 
        for bvh_file in all_bvh_files:
            filename = bvh_file.split("/")[-1].split(".")[0]
            bvh_dict = bvh.load(bvh_file)
            bvh_pos = bvh_dict["positions"]
            index_map = [beat_MM.index(i) for i in beat2smpl_new_MM]
            pos_bvh_order = []
            for m in range(bvh_pos.shape[0]):
                pos_list = []
                for i in index_map:
                    pos_list.append(bvh_pos[m][i])
                pos_bvh_order.append(np.array(pos_list))
            pos_bvh_order = np.array(pos_bvh_order)[:50]
            np.save(str(bvh_save_path / f"{filename}_MM.npy"), arr=pos_bvh_order)


############################### PyMO Utils #####################################

if __name__ == "__main__":
    
    
    """
    1. raw to transfer2target: 
        fps downsample, con - emo division, normalization, jlist selection
    2. cutting it to slice of 100 frames
    3. bvh file creation:
        unnormalize, combine con and emo, result2target_vis
    """
    # import torch
    # from PyMO.pymo.parsers import BVHParser
    # from PyMO.pymo.viz_tools import *
    # from PyMO.pymo.preprocessing import *
    # from PyMO.pymo.features import *
    # from PyMO.pymo.writers import *
    
    # src_bvh = "/home/kchhatre/Work/code/disentangled-s2g/data/beat-rawdata-eng/beat_rawdata_english/1/1_wayne_0_66_66.bvh"
    # bvh_files = [
    #     "/home/kchhatre/Work/code/disentangled-s2g/data/beat-rawdata-eng/beat_rawdata_english/1/1_wayne_0_66_66.bvh",
    #     "/home/kchhatre/Work/code/disentangled-s2g/data/beat-rawdata-eng/beat_rawdata_english/1/1_wayne_0_72_72.bvh"
    # ]
    # original_fps = 120
    # targest_fps = 25
    # dwnsmpl_factor = "f4"  # "f5"
    # con = "audio_sync_j_v0"
    # emo = "remaining_j_v0"
    # combined = "combined_v0"
    # tmp_savepath = Path("/home/kchhatre/Work/code/disentangled-s2g/tests/tmp_results/")
    
    # cfg = {
    #     "DATA_PARAM": { 
    #         "Bvh": {
    #             "con_emo_div": {
    #                 "use": True ,
    #                 "con": "audio_sync_j_v0",
    #                 "emo": "remaining_j_v0",
    #                 "combined": "combined_v0"
    #             },
    #             "fps": 25, # 30, 60, 120 (original recording fps 121)
    #             "joint_name_list": "joint_name_list_27" # audio_sync_j_v0 remaining_j_v0 joint_name_list_225, joint_name_list_186, joint_name_list_27, joint_name_list_27_v3, spine_neck_141, torso_fingers 
    #             } 
    #         }
    # }

    # transfer2target(bvh_file=src_bvh, bvh_save_path=tmp_savepath, config=cfg)
    
    # mean_con = np.load(tmp_savepath / "mean_con.npy")
    # std_con = np.load(tmp_savepath / "std_con.npy")
    # mean_emo = np.load(tmp_savepath / "mean_emo.npy")
    # std_emo = np.load(tmp_savepath / "std_emo.npy")
    # mean_combined = np.load(tmp_savepath / "mean_combined.npy")
    # std_combined = np.load(tmp_savepath / "std_combined.npy")
    
    # all_data = {} 
    # all_data["wayne"] = {}
    # all_data["wayne"]["0_66_66"] = {}   
    # for file in tqdm(glob.glob(str(tmp_savepath) + "/*.bvh"), 
    #                 desc="[DIFF] (1/3) Loading preprocessed BVHs", leave=False):
    #     string_id = file.split("/")[-1].split(".")[0]
    #     num, actor, base_take = string_id.split("_")[0], \
    #                     string_id.split("_")[1], \
    #                     "_".join(string_id.split("_")[2:])
    #     if base_take.split("_")[-1] in ["con", "emo", "combined"]:
    #         take = "_".join(base_take.split("_")[:-1])
    #         take_type = base_take.split("_")[-1]
    #     else:
    #         take = base_take
    #         take_type = ""
    #     # print(f"num: {num}, actor: {actor}, take: {take}, take_type: {take_type}, string_id: {string_id}, base_take: {base_take}")
    #     entry = "diff_bvh" + "_" + take_type if take_type != "" else "diff_bvh"
        
    #     all_data[actor][take][entry] = []
    #     with open(file, "r") as f:
    #         for j, line in enumerate(f.readlines()):
    #             pose_data = np.fromstring(line, dtype=float, sep=" ") # eg. Li (27,)
    #             all_data[actor][take][entry].append(pose_data)
    #     all_data[actor][take][entry] = torch.tensor(np.stack(all_data[actor][take][entry]))
    #     print(f"entry: {entry}, shape: {all_data[actor][take][entry].shape}")
    
    # i = actor
    # j = take
    # verbose = True
    # all_data[i][j]["diff_bvh_con_n"] = normalize(all_data[i][j]["diff_bvh_con"], mean_con, std_con, verbose)
    # all_data[i][j]["diff_bvh_emo_n"] = normalize(all_data[i][j]["diff_bvh_emo"], mean_emo, std_emo, verbose)
    # all_data[i][j]["diff_bvh_combined_n"] = normalize(all_data[i][j]["diff_bvh_combined"], mean_combined, std_combined, verbose)
    # print(f"diff_bvh_con_n: {all_data[i][j]['diff_bvh_con_n'].shape}")
    # print(f"diff_bvh_emo_n: {all_data[i][j]['diff_bvh_emo_n'].shape}")
    # print(f"diff_bvh_combined_n: {all_data[i][j]['diff_bvh_combined_n'].shape}")
    
    # print(all_data[i][j]["diff_bvh_con"])
    # print(mean_con)
    # print(std_con)
    # print(all_data[i][j]['diff_bvh_con_n'])
    
    # print("shapes: ", (all_data[i][j]["diff_bvh_con"].shape, mean_con.shape, std_con.shape))
    
    # def minmax(val_list):
    #     min_val = min(val_list)
    #     max_val = max(val_list)
    #     min_index = val_list.index(min_val)
    #     max_index = val_list.index(max_val)
    #     return (min_val, max_val, min_index, max_index)
    
    # print(f"minmax of diff_bvh_con: {minmax(all_data[i][j]['diff_bvh_con'].flatten().tolist())}")
    # print(f"minmax of diff_bvh_con_n: {minmax(all_data[i][j]['diff_bvh_con_n'].flatten().tolist())}")
    
    # pymo_pipeline(bvh_files, mocap_type, bvh_fps, verbose=False)
        
    # Old Debugging 
    # bvh_path = "/home/kchhatre/Work/code/disentangled-s2g/data/beat-rawdata-eng/beat_rawdata_english/1/1_wayne_1_10_10.bvh"
    # bvh_save_path = Path("/home/kchhatre/Work/code/disentangled-s2g/tests/tmp_results/")
    # testpath = "/home/kchhatre/Work/code/disentangled-s2g/data/BEAT-processed/processed-all-modalities/bvh-fps25-joint_name_list_27-div"
    # config = {
    #     "DATA_PARAM": { 
    #         "Bvh": {
    #             "con_emo_div": {
    #                 "use": True ,
    #                 "con": "audio_sync_j_v0",
    #                 "emo": "remaining_j_v0",
    #                 "combined": "combined_v0"
    #             },
    #             "fps": 25, # 30, 60, 120 (original recording fps 121)
    #             "joint_name_list": "joint_name_list_27" # audio_sync_j_v0 remaining_j_v0 joint_name_list_225, joint_name_list_186, joint_name_list_27, joint_name_list_27_v3, spine_neck_141, torso_fingers 
    #             } 
    #         }
    # }
    
    # #                                                                     Final@25FPS: (BEAT 15FPS DIMS BELOW)
    # file = str(bvh_save_path) + "/1_wayne_1_10_10.bvh"                  # 1_wayne_1_10_10.bvh (15333, 186)
    # con = str(bvh_save_path) + "/1_wayne_1_10_10_con.bvh"               # 1_wayne_1_10_10_con.bvh (15333, 24)
    # emo = str(bvh_save_path) + "/1_wayne_1_10_10_emo.bvh"               # 1_wayne_1_10_10_emo.bvh (15333, 30)
    # combined = str(bvh_save_path) + "/1_wayne_1_10_10_combined.bvh"     # 1_wayne_1_10_10_combined.bvh (15333, 54)
    
    # t_c = testpath + "/6_carla_0_13_13_con.bvh"
    # t_e = testpath + "/6_carla_0_13_13_emo.bvh"
    # t_combined = testpath + "/6_carla_0_13_13_combined.bvh"
    
    # b_file = str(bvh_save_path) + "/1_wayne_1_10_10_beat30fps.bvh" 
    
    # div = [con, emo, combined]
    # nondiv = [file]
    # testfiles = [t_c, t_e, t_combined]
    # beat_fps_file =[b_file]
    # import os # lazy import
    
    # # bvh_path = "/home/kchhatre/Work/code/disentangled-s2g/data/BEAT-processed/processed-all-modalities/bvh-fps25-audio_sync_j_v0-div-f4/1_wayne_0_9_9_emo.bvh"
    # # emo  = str(bvh_save_path) + "/1_wayne_0_9_9_emo.bvh"
    # # div = [emo]
    # # d_mean = "/home/kchhatre/Work/code/disentangled-s2g/data/BEAT-processed/processed-all-modalities/bvh-fps25-audio_sync_j_v0-div-f4/mean_emo.npy"
    # # d_std ="/home/kchhatre/Work/code/disentangled-s2g/data/BEAT-processed/processed-all-modalities/bvh-fps25-audio_sync_j_v0-div-f4/std_emo.npy"
    
    # for file in div:
    # # # for file in testfiles:
    # # # for file in beat_fps_file:
    # # # for file in nondiv:
    
    #     if not os.path.isfile(file):
    #         print("Processing file")
    #         transfer2target(bvh_path, bvh_save_path, config)
    #         print("Done processing file")
    #     else:
    #         print("File already exists")
            
    #     all_poses = []
    #     with open(file, "r") as f:
    #         for j, line in enumerate(f.readlines()):
    #             pose_data = np.fromstring(line, dtype=float, sep=" ")
    #             all_poses.append(pose_data)
    #     all_poses = np.array(all_poses)
    #     print(file.split("/")[-1], all_poses.shape) 
    
    # # for 186 & 25: FPS -> (15333, 186) (FPS related, joints related)
    # # for 186 & 30: FPS -> (15333, 186)
    # # for 186 & 60: FPS -> (25554, 186)
    # # for 186 & 120: FPS -> (38331, 186)
    # # for 186 & 121: FPS -> (76662, 186)
    
    # # 27, 25 FPS: (15333, 27)
    # # 225, 25 FPS: (15333, 225)
    
    # # con, 25 fps: (15333, 24)
    # # emo, 25 fps: (15333, 30)
    # # combined, 25 fps: (15333, 54)
    
    # # BEAT w 15 fps: (8518, 27)
    # # BEAT w 30 fps: (15333, 27)
    # # BEAT w 60 fps: (25554, 27)
    # # BEAT w 120 fps: (38331, 27)
    
    # # Debug mean, std func
    # # ms_bvh_save_path = Path("/home/kchhatre/Work/code/disentangled-s2g/data/BEAT-processed/processed-all-modalities/bvh-fps25-audio_sync_j_v0-div-f4")
    # # ms_bvh_save_path = Path("/home/kchhatre/Work/code/disentangled-s2g/data/BEAT-processed/processed-all-modalities/bvh-fps25-audio_sync_j_v0-div-f5")
    # # get_mean_pose(ms_bvh_save_path, verbose=True)
    
    # # Debug normalize func
    # mean = np.load(d_mean)
    # std = np.load(d_std)
    # normalized = normalize(all_poses, mean, std, verbose=True)
    # print("Normalized shape", normalized.shape, "original shape", all_poses.shape)
    
    # # TODO: visulaize audio_sync_j_v0 and remaining_j_v0 together and separately
    
    
################################################################################

