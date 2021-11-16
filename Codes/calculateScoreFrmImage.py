# importing libraries
import numpy as np
import cv2
import time
from matplotlib import pyplot as plt
import os
from natsort import natsorted


tp = 0
tn = 0
fp = 0
fn = 0

root_path = "/home/fatih/phd/tik3/bg_model/"

inx = 1

mog = "mog.mp4"
mog2 = "mog2.mp4"
knn = "knn.mp4"

h = 1080
w = 1920

gt_img = os.listdir(root_path+"gt_binarymask/")
gmg_img = os.listdir(root_path+"gmg/")

gt_img = natsorted(gt_img)
gmg_img = natsorted(gmg_img)

for img in gt_img:
    gt_inst = cv2.imread(root_path+"gt_binarymask/"+img)
    gmg_inst = cv2.imread(root_path+"gmg/"+img)

    """
    cv2.imshow("GT",gt_frame)
    cv2.imshow("GMG",gmg_frame)
    """        
    for y in range(h):
        for x in range(w):
            if gt_inst[y,x][0] == 0 and gmg_inst[y,x][0] == 0:
                tn += 1
            if gt_inst[y,x][0] == 0 and gmg_inst[y,x][0] != 0:
                fp += 1
            if gt_inst[y,x][0] != 0 and gmg_inst[y,x][0] == 0:    
                fn += 1
            if gt_inst[y,x][0] != 0 and gmg_inst[y,x][0] != 0:
                tp += 1                                    
    #print("Frame {}".format(inx))
    print(inx)
    inx += 1
    print("tp: {}, tn: {}, fp: {}, fn: {}".format(tp,tn,fp,fn))
    #if cv2.waitKey(30) & 0xFF == ord('q'):
        #break

print("tp: {}, tn: {}, fp: {}, fn: {}, acc: {}, prec: {}".format(tp,tn,fp,fn,(tp+tn)/(tp+tn+fp+fn),tp/(tp+fp)))



accuracy_gmg = 0
accuracy_mog = 0
accuracy_mog2 = 0
accuracy_knn = 0