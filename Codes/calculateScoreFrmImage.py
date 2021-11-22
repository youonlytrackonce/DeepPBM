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
#gmg_img = os.listdir(root_path+"mog/")

gt_img = natsorted(gt_img)
#gmg_img = natsorted(gmg_img)

for img in gt_img:
    gt_inst = cv2.imread(root_path+"gt_binarymask/"+img)
    gmg_inst = cv2.imread(root_path+"deeppbm_mask/"+img)

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

# tp: 99254130, tn: 3290320450, fp: 101645417, fn: 241260003, acc: 0.9081293349194102, prec: 0.4940485505425256 gmg
# tp: 163401818, tn: 3100405802, fp: 291560065, fn: 177112315, acc: 0.8744340545696159, prec: 0.3591549624389083 mog2
# tp: 108046339, tn: 3324923121, fp: 67042746, fn: 232467794, acc: 0.9197556209276406, prec: 0.6170935155666614 mog
# tp: 164847644, tn: 3168073440, fp: 223892427, fn: 175666489, acc: 0.892950821973594, prec: 0.42405621724548176 knn
# tp: 162973630, tn: 3019134152, fp: 372831715, fn: 177540503, acc: 0.8525451662165637, prec: 0.3041657413850547 deeppbm