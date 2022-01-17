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

root_path = "/home/ubuntu/phd/DeepPBM/experiment_bmc/Video_002/"

inx = 1


h = 288
w = 352

gt_img = os.listdir(root_path + "gt_img/")
gt_img = natsorted(gt_img)

resnet_mask = os.listdir(root_path + "resnet_mask/")
resnet_mask = natsorted(resnet_mask)

vanilla_mask = os.listdir(root_path + "vanilla_mask/")
vanilla_mask = natsorted(vanilla_mask)


for ii in range(len(gt_img)):
    gt_inst = cv2.imread(root_path+"gt_img/" + gt_img[ii])
    resnet_inst = cv2.imread(root_path+"resnet_mask/" + resnet_mask[ii])
    vanilla_inst = cv2.imread(root_path + "vanilla_mask/" + vanilla_mask[ii])

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

    print(ii)


recall_resnet = tp_resnet / (tp_resnet + fn_resnet)
print('recall resnet: {}'.format(recall_resnet))
specificity_resnet = tn_resnet / (tn_resnet + fp_resnet)
print('specificity resnet: {}'.format(specificity_resnet))
FPR_resnet = fp_resnet / (fp_resnet + tn_resnet)
print('FPR resnet: {}'.format(FPR_resnet))
FNR_resnet = fn_resnet / (tp_resnet + fn_resnet)
print('FNR resnet: {}'.format(FNR_resnet))
PBC_resnet = 100.0 * (fn_resnet + fp_resnet) / (tp_resnet + fp_resnet + fn_resnet + tn_resnet)
print('PBC resnet: {}'.format(PBC_resnet))
precision_resnet = tp_resnet / (tp_resnet + fp_resnet)
print('precision resnet: {}'.format(precision_resnet))
FMeasure_resnet = 2.0 * (recall_resnet * precision_resnet) / (recall_resnet + precision_resnet)
print('FMeasure resnet: {}'.format(FMeasure_resnet))

recall_vanilla = tp_vanilla / (tp_vanilla + fn_vanilla)
print('recall vanilla: {}'.format(recall_vanilla))
specificity_vanilla = tn_vanilla / (tn_vanilla + fp_vanilla)
print('specificity vanilla: {}'.format(specificity_vanilla))
FPR_vanilla = fp_vanilla / (fp_vanilla + tn_vanilla)
print('FPR vanilla: {}'.format(FPR_vanilla))
FNR_vanilla = fn_vanilla / (tp_vanilla + fn_vanilla)
print('FNR vanilla: {}'.format(FNR_vanilla))
PBC_vanilla = 100.0 * (fn_vanilla + fp_vanilla) / (tp_vanilla + fp_vanilla + fn_vanilla + tn_vanilla)
print('PBC vanilla: {}'.format(PBC_vanilla))
precision_vanilla = tp_vanilla / (tp_vanilla + fp_vanilla)
print('precision vanilla: {}'.format(precision_vanilla))
FMeasure_vanilla = 2.0 * (recall_vanilla * precision_vanilla) / (recall_vanilla + precision_vanilla)
print('FMeasure vanilla: {}'.format(FMeasure_vanilla))

