##################################################
#This file for conducting some small ablation experiments on CFP dataset
##################################################
############################ NN general packages
import torch
import torch.nn as nn
import numpy as np
import shutil
############################ Operation packages
import math
import struct
from Subnet_def import *


def load_feature(feature_path):
    feature = list()
    with open(feature_path, 'rb') as f:
        feat_num, feat_size = struct.unpack('ii', f.read(8))                    # 2 int variables occupy 8 Bytes
        for i in range(feat_num):
            feat = np.array(struct.unpack('f'*feat_size, f.read(4*feat_size)))  # 256 float variables occupy 256*4  Bytes
            feature.append(feat)
    return feature


if __name__ == '__main__':
    frontal_feature_path = '../../data/plugin/frontal_features.bin'             #feature extracted from any face recognition model, like ArcFace, CosFace, ...
    profile_feature_path = '../../data/plugin/profile_features.bin'

    frontal_features = np.vstack(load_feature(frontal_feature_path))
    profile_features = np.vstack(load_feature(profile_feature_path))

    frontal_angles = []
    profile_angles = []
    with open('../../data/pose_estimation', 'r') as fpose:                        #prior pose labels path
        tmp_cnt = 0
        for line in fpose:
            line_split = line.strip().split()                                     # CFP dataset has 10 frontal faces and 4  profiles faces 
            if tmp_count % 14 < 10:
                frontal_angles.append(float(line_split[2]))
            else:
                profile_angles.append(float(line_split[2]))
            tmp_count = tmp_count + 1
    frontal_angles = np.vstack(frontal_angles)
    profile_angles = np.vstack(profile_angles)

    model = Res_Subnet(feat_dim=256)
    model.cuda()
    paras_model = torch.load('../../data/plugin/plugin_subset.pth')                # load training parameters
    model.load_state_dict(paras_model['state_dict'])

    model.eval()
    expected_frontal_path = './expected_frontal_features.bin'                      # the result of expected frontal feature
    feat_dim = 256
    data_num = frontal_features.shape[0]
    with open(expected_frontal_path, 'wb') as fout:
        fout.write(struct.pack('ii', data_num, feat_dim))
        for i in range(data_num):
            feat_vec = frontal_features[i, :].reshape([1, -1])                        # 1 row vector
            angle = np.zeros([1, 1])
            angle[0,0] = Gating_Control(frontal_angles[i, :])
            feat_vec = torch.autograd.Variable(torch.from_numpy(feat_vec.astype(np.float32)), volatile=True).cuda()
            angle = torch.autograd.Variable(torch.from_numpy(angle.astype(np.float32)), volatile=True).cuda()
            result = model(feat_vec, angle)
            out_result = result.cpu().data.numpy()
            feat_num  = result.size(0)
            for j in range(feat_num):
                fout.write(struct.pack('f'*feat_dim, *tuple(out_result[j,:])))

    data_num = profile_feats.shape[0]
    expected_profile_path = './expected_profile_features.bin'                      # the result of expected profile feature
    with open(expected_profile_path, 'wb') as fout:
        fout.write(struct.pack('ii', data_num, feat_dim))
        for i in range(data_num):
            feat_vec = profile_features[i, :].reshape([1, -1])
            angle = np.zeros([1, 1])
            angle[0,0] = Gating_Control(profile_angles[i, :])
            feat_vec = torch.autograd.Variable(torch.from_numpy(feat_vec.astype(np.float32)), volatile=True).cuda()
            angle = torch.autograd.Variable(torch.from_numpy(angle.astype(np.float32)), volatile=True).cuda()
            result = model(feat_vec, angle)
            out_result = result.cpu().data.numpy()
            feat_num  = result.size(0)
            for j in range(feat_num):
                fout.write(struct.pack('f'*feat_dim, *tuple(out_result[j,:])))
