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

videoNum = 8

path_gt = '/home/ubuntu/phd/DeepPBM/experiment_bmc/Video_00{}/gt_img/'.format(videoNum)
resnet_path = '/home/ubuntu/phd/DeepPBM/experiment_bmc/Video_00{}/resnet_epoch140_batch4_z1_lr0.01_gtx1080/'.format(videoNum)
vanilla_path = '/home/ubuntu/phd/DeepPBM/experiment_bmc/Video_00{}/vanilla_epoch100_batch4_z2_lr0.001_gtx1080/'.format(videoNum)

gmg_path = '/home/ubuntu/phd/DeepPBM/experiment_bmc/Video_00{}/gmg_mask/'.format(videoNum)
knn_path = '/home/ubuntu/phd/DeepPBM/experiment_bmc/Video_00{}/knn_mask/'.format(videoNum)
mog2_path = '/home/ubuntu/phd/DeepPBM/experiment_bmc/Video_00{}/mog2_mask/'.format(videoNum)

target_resnet = '/home/ubuntu/phd/DeepPBM/experiment_bmc/Video_00{}/resnet/'.format(videoNum)
target_vanilla = '/home/ubuntu/phd/DeepPBM/experiment_bmc/Video_00{}/vanilla/'.format(videoNum)

target_gmg = '/home/ubuntu/phd/DeepPBM/experiment_bmc/Video_00{}/gmg_mask_reduce/'.format(videoNum)
target_knn = '/home/ubuntu/phd/DeepPBM/experiment_bmc/Video_00{}/knn_mask_reduce/'.format(videoNum)
target_mog2 = '/home/ubuntu/phd/DeepPBM/experiment_bmc/Video_00{}/mog2_mask_reduce/'.format(videoNum)

gt_list = os.listdir(path_gt)
gt_list = natsorted(gt_list)


for gt in gt_list:
    name = gt.split('.')[0]
    name = name.split('_')[1]
    name = name.zfill(6)
    int_name = int(name)
    shutil.copyfile(resnet_path + name + '.jpg', target_resnet + str(int_name) + '.jpg')
    shutil.copyfile(vanilla_path + str(int_name) + '.jpg', target_vanilla + str(int_name) + '.jpg')
    shutil.copyfile(gmg_path + str(int_name) + '.jpg', target_gmg + str(int_name) + '.jpg')
    shutil.copyfile(knn_path + str(int_name) + '.jpg', target_knn + str(int_name) + '.jpg')
    shutil.copyfile(mog2_path + str(int_name) + '.jpg', target_mog2 + str(int_name) + '.jpg')
