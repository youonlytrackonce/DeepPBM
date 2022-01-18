# importing libraries
import numpy as np
import cv2
import time
import os
from natsort import natsorted

tp_resnet = 0
tn_resnet = 0
fp_resnet = 0
fn_resnet = 0

tp_vanilla = 0
tn_vanilla = 0
fp_vanilla = 0
fn_vanilla = 0

tp_gmg = 0
tn_gmg = 0
fp_gmg = 0
fn_gmg = 0

tp_knn = 0
tn_knn = 0
fp_knn = 0
fn_knn = 0

tp_mog2 = 0
tn_mog2 = 0
fp_mog2 = 0
fn_mog2 = 0

video = 8

root_path = "/home/ubuntu/phd/DeepPBM/experiment_bmc/Video_00{}/".format(video)

inx = 1

h = 288
w = 352

gt_img = os.listdir(root_path + "gt_img/")
gt_img = natsorted(gt_img)

resnet_mask = os.listdir(root_path + "resnet_mask/")
resnet_mask = natsorted(resnet_mask)

vanilla_mask = os.listdir(root_path + "vanilla_mask/")
vanilla_mask = natsorted(vanilla_mask)

gmg_mask = os.listdir(root_path + "gmg_mask_reduce/")
gmg_mask = natsorted(gmg_mask)

knn_mask = os.listdir(root_path + "knn_mask_reduce/")
knn_mask = natsorted(knn_mask)

mog2_mask = os.listdir(root_path + "mog2_mask_reduce/")
mog2_mask = natsorted(mog2_mask)

for ii in range(len(gt_img)):
    gt_inst = cv2.imread(root_path + "gt_img/" + gt_img[ii])
    resnet_inst = cv2.imread(root_path + "resnet_mask/" + resnet_mask[ii])
    vanilla_inst = cv2.imread(root_path + "vanilla_mask/" + vanilla_mask[ii])

    gmg_inst = cv2.imread(root_path + "gmg_mask_reduce/" + gmg_mask[ii])
    knn_inst = cv2.imread(root_path + "knn_mask_reduce/" + knn_mask[ii])
    mog2_inst = cv2.imread(root_path + "mog2_mask_reduce/" + mog2_mask[ii])

    for y in range(h):
        for x in range(w):
            if gt_inst[y, x][0] == 0 and resnet_inst[y, x][0] == 0:
                tn_resnet += 1
            if gt_inst[y, x][0] == 0 and resnet_inst[y, x][0] != 0:
                fp_resnet += 1
            if gt_inst[y, x][0] != 0 and resnet_inst[y, x][0] == 0:
                fn_resnet += 1
            if gt_inst[y, x][0] != 0 and resnet_inst[y, x][0] != 0:
                tp_resnet += 1

            if gt_inst[y, x][0] == 0 and vanilla_inst[y, x][0] == 0:
                tn_vanilla += 1
            if gt_inst[y, x][0] == 0 and vanilla_inst[y, x][0] != 0:
                fp_vanilla += 1
            if gt_inst[y, x][0] != 0 and vanilla_inst[y, x][0] == 0:
                fn_vanilla += 1
            if gt_inst[y, x][0] != 0 and vanilla_inst[y, x][0] != 0:
                tp_vanilla += 1

            if gt_inst[y, x][0] == 0 and gmg_inst[y, x][0] == 0:
                tn_gmg += 1
            if gt_inst[y, x][0] == 0 and gmg_inst[y, x][0] != 0:
                fp_gmg += 1
            if gt_inst[y, x][0] != 0 and gmg_inst[y, x][0] == 0:
                fn_gmg += 1
            if gt_inst[y, x][0] != 0 and gmg_inst[y, x][0] != 0:
                tp_gmg += 1

            if gt_inst[y, x][0] == 0 and knn_inst[y, x][0] == 0:
                tn_knn += 1
            if gt_inst[y, x][0] == 0 and knn_inst[y, x][0] != 0:
                fp_knn += 1
            if gt_inst[y, x][0] != 0 and knn_inst[y, x][0] == 0:
                fn_knn += 1
            if gt_inst[y, x][0] != 0 and knn_inst[y, x][0] != 0:
                tp_knn += 1

            if gt_inst[y, x][0] == 0 and mog2_inst[y, x][0] == 0:
                tn_mog2 += 1
            if gt_inst[y, x][0] == 0 and mog2_inst[y, x][0] != 0:
                fp_mog2 += 1
            if gt_inst[y, x][0] != 0 and mog2_inst[y, x][0] == 0:
                fn_mog2 += 1
            if gt_inst[y, x][0] != 0 and mog2_inst[y, x][0] != 0:
                tp_mog2 += 1
    print(ii)

recall_resnet = tp_resnet / (tp_resnet + fn_resnet)
print('recall resnet: {}'.format(recall_resnet))
precision_resnet = tp_resnet / (tp_resnet + fp_resnet)
print('precision resnet: {}'.format(precision_resnet))
FMeasure_resnet = 2.0 * (recall_resnet * precision_resnet) / (recall_resnet + precision_resnet)
print('FMeasure resnet: {}'.format(FMeasure_resnet))

recall_vanilla = tp_vanilla / (tp_vanilla + fn_vanilla)
print('recall vanilla: {}'.format(recall_vanilla))
precision_vanilla = tp_vanilla / (tp_vanilla + fp_vanilla)
print('precision vanilla: {}'.format(precision_vanilla))
FMeasure_vanilla = 2.0 * (recall_vanilla * precision_vanilla) / (recall_vanilla + precision_vanilla)
print('FMeasure vanilla: {}'.format(FMeasure_vanilla))

recall_gmg = tp_gmg / (tp_gmg + fn_gmg)
print('recall gmg: {}'.format(recall_gmg))
precision_gmg = tp_gmg / (tp_gmg + fp_gmg)
print('precision gmg: {}'.format(precision_gmg))
FMeasure_gmg = 2.0 * (recall_gmg * precision_gmg) / (recall_gmg + precision_gmg)
print('FMeasure gmg: {}'.format(FMeasure_gmg))

recall_knn = tp_gmg / (tp_knn + fn_knn)
print('recall knn: {}'.format(recall_knn))
precision_knn = tp_knn / (tp_knn + fp_knn)
print('precision knn: {}'.format(precision_knn))
FMeasure_knn = 2.0 * (recall_knn * precision_knn) / (recall_knn + precision_knn)
print('FMeasure knn: {}'.format(FMeasure_knn))

recall_mog2 = tp_mog2 / (tp_mog2 + fn_mog2)
print('recall mog2: {}'.format(recall_mog2))
precision_mog2 = tp_mog2 / (tp_mog2 + fp_mog2)
print('precision mog2: {}'.format(precision_mog2))
FMeasure_mog2 = 2.0 * (recall_mog2 * precision_mog2) / (recall_mog2 + precision_mog2)
print('FMeasure mog2: {}'.format(FMeasure_mog2))
