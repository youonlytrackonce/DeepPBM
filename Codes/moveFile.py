import cv2 as cv
import numpy as np
import math
from PIL import Image, ImageChops
from numpy import asarray
import os
import copy
import time
from natsort import natsorted
import shutil

videoNum = 2

path_gt = '/home/ubuntu/phd/DeepPBM/experiment_bmc/Video_00{}/gt_img/'.format(videoNum)
resnet_path = '/home/ubuntu/phd/DeepPBM/experiment_bmc/Video_00{}/resnet_epoch200_batch4_z1_lr0.01/'.format(videoNum)
vanilla_path = '/home/ubuntu/phd/DeepPBM/experiment_bmc/Video_00{}/vanilla_epoch200_batch4_z1_lr0.001/'.format(videoNum)

target_resnet = '/home/ubuntu/phd/DeepPBM/experiment_bmc/Video_00{}/resnet/'.format(videoNum)
target_vanilla = '/home/ubuntu/phd/DeepPBM/experiment_bmc/Video_00{}/vanilla/'.format(videoNum)

gt_list = os.listdir(path_gt)
gt_list = natsorted(gt_list)


for gt in gt_list:
    name = gt.split('.')[0]
    name = name.split('_')[1]
    name = name.zfill(6)
    int_name = int(name)
    shutil.copyfile(resnet_path + name + '.jpg', target_resnet + str(int_name) + '.jpg')
    shutil.copyfile(vanilla_path + str(int_name) + '.jpg', target_vanilla + str(int_name) + '.jpg')