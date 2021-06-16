############################ NN general packages
import torch
import torch.nn as nn
import numpy as np
import argparse
import shutil
############################ Operation packages
import math
import random
import struct
from Subnet_def import *



## Default parameters
parser = argparse.ArgumentParser(description='Default Parameter Control')
parser.add_argument('--rsl', '--residual_sample_list', default='../../data/plugin/residual_sample_list.txt', type=str, help='the path of the residual sample list')
parser.add_argument('--cf', '--clean_feature', default='../../data/plugin/clean_feature.bin', type=str, help='the path of clean feature')
parser.add_argument('--bs', '--batch_size', default=256, type=int, help='batch size-256')
parser.add_argument('--fs', '--feature_size', default=256, type=int, help=' the length of feature vector-256')
parser.add_argument('--its', '--iterations', default=10000, type=int,  help=' the number of total iterations ')
parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float, help='learning rate')
parser.add_argument('--mot', '--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--wtd', '--weight_decay', default=1e-4, type=float,  help='weight-1e-4)')



def load_sample(sample_path):
    sample_feature = dict()
    with open(sample_path) as f:
        for id, line in enumerate(f):
            text = line.strip().split()
            img_name, angle, img_id = text[0],float(text[1]), id                #load information
            indiv = img_name.split('/')[1]
            if indiv not in sample_feature:
                sample_feature[indiv] = (list(), list())                        #prepare
            angle = abs(angle)
            if angle <  15:                                                     #selete samples with large angle
                sample_feature[indiv][0].append((img_name, angle, img_id))
            elif angle >  45:
                sample_feature[indiv][1].append((img_name, angle, img_id))
    pop_list = []
    for key in sample_feature:                                                  #delete samples without large angle
        if len(sample_feature[key][0]) == 0 or len(sample_feature[key][1]) == 0:
            pop_list.append(key)
    for key in pop_list:
        sample_feature.pop(key)
    return sample_feature


def load_feature(feature_path):
    feature = list()
    with open(feature_path, 'rb') as f:
        feat_num, feat_size = struct.unpack('ii', f.read(8))                     # 2 int variables occupy 8 Bytes
        for i in range(feat_num):
            feat = np.array(struct.unpack('f'*feat_size, f.read(4*feat_size)))  # 256 float variables occupy 256*4  Bytes
            feature.append(feat)
    return feature



def load_batch(residual_sample, cl_feat_map, batch_size, feature_size):
    batch_train = np.zeros([batch_size, feature_size])
    batch_target = np.zeros([batch_size, feature_size])
    batch_angle = np.zeros([batch_size, 1])

    keys = residual_sample.keys()                                                # keys are names/identifies
    for i in range(batch_size):
        current_key = random.sample(keys, 1)[0]                                  # randomly pick one name  ([0]:list->str)
        frontal_set = residual_sample[current_key][0]
        profile_set = residual_sample[current_key][1]
        frontals_id = [a[2] for a in frontal_set]                                # data format:  [0]path&name, [1]angle, [2]id
        frontals_feat = cl_feat_map[frontals_id, :]                            # clean id-th feature
        profile_sele = random.sample(profile_set, 1)                             # randomly pick one profile face from this identity profile faces
        batch_train[i, :] = cl_feat_map[profile_sele[0][2], :]                 # profile feature
        batch_target[i, :] =  np.mean(frontals_feat, axis = 0)                 # frontal face feature (column aver): 256 vector
        batch_angle[i, :] = Gating_Control(profile_selec[0][1])                # angle -> sin
    return batch_train, batch_target, batch_angle


class AverValue():                                                               #Compute average value
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt



if __name__ == '__main__':
    args = parser.parse_args()
    

    residual_sample_list_path = args.residual_sample_list
    residual_sample = load_sample(residual_sample_list_path)
    clean_feature_path = args.clean_feature
    clean_feature = load_feat(clean_feature_path)

    cl_feat_map = np.vstack(clean_feature)                              #creat clean feature map
    model = Res_Subnet(feat_size=256)
    model.cuda()
    metric = nn.MSELoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.mot, weight_decay=args.wtd)


    model.train()
    losses = AverValue()
    for iter in range(args.its):
        batch_train, batch_target, batch_sin = load_batch(residual_sample, cl_feat_map, args.bs, args.fs)                 #batch size
        batch_train = torch.autograd.Variable(torch.from_numpy(batch_train.astype(np.float32))).cuda()
        batch_target = torch.autograd.Variable(torch.from_numpy(batch_target.astype(np.float32))).cuda()
        batch_sin   = torch.autograd.Variable(torch.from_numpy(batch_sin.astype(np.float32))).cuda()

        res_feature = model(batch_train, batch_sin)                                          # MSE and mean error
        loss = metric(res_feature, batch_target)
        losses.update(loss.data[0], loss.size(0))


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter % 100 == 0:                                                                    # print frequency :100 
            print('Training process: [{0}/{1}]- Loss: {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   iter, args.iters, loss=losses))

    torch.save({'state_dict': model.state_dict()}, '../../data/plugin/plugin_subset.pth')                           # only save params
