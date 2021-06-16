import torch
import torch.nn as nn
import math



def Gating_Control(angle):                           #Subnet1: The gating control function--|sin()|
    Gating_Control = abs(sin(angle))
    return Gating_Control

class Res_Subnet(nn.Module):                         #Subnet2: The residual Resnet with 2 FC+ReLU
    def __init__(self, feat_size):
        super(Res_Subnet, self).__init__()           # Res_Subnet can inherit nn.Moudle
        self.fc1 = nn.Linear(feat_size, feat_size)   # keep the length of i/o feature vector fixed
        self.fc2 = nn.Linear(feat_size, feat_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input, angle):                 # learning residual component res_p
        res_p = self.fc1(input)
        res_p = self.relu(res_p)
        res_p = self.fc2(res_p)
        res_p = self.relu(res_p)

        angle = angle.view(angle.size(0),1)
        angle = angle.expand_as(res_p)

        out_f = angle * res_p + input                # angle has been changed to |sin() before|
        return out_f                                 # out_f -- expected frontal feature