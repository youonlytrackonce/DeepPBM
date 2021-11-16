# importing libraries
import numpy as np
import cv2
import time
from matplotlib import pyplot as plt


tp = 0
tn = 0
fp = 0
fn = 0

root_path = "/home/fatih/phd/tik3/bg_model/"

inx = 1

mog = "mog.mp4"
mog2 = "mog2.mp4"
knn = "knn.mp4"

gt = cv2.VideoCapture(root_path+"gt_binarymask.mp4")
gmg = cv2.VideoCapture(root_path+"gmg.mp4")

# Check if camera opened successfully
if (gt.isOpened()== False):
	print("Error opening gt video stream or file")

if (gmg.isOpened()== False):
	print("Error opening gmg video stream or file")

while(gt.isOpened() and gmg.isOpened()):
    gt_ret, gt_frame = gt.read()
    h = gt_frame.shape[0]
    w = gt_frame.shape[1]
    gmg_ret, gmg_frame = gmg.read()
    if gmg_ret and gt_ret:
        """
        cv2.imshow("GT",gt_frame)
        cv2.imshow("GMG",gmg_frame)
        """        
        for y in range(h):
            for x in range(w):
                print("gt: {}".format(gt_frame[y,x][0]))
                print("gmg: {}".format(gmg_frame[y,x][0]))
                if gt_frame[y,x][0] == 0 and gmg_frame[y,x][0] == 0:
                    tn += 1
                if gt_frame[y,x][0] == 0 and gmg_frame[y,x][0] != 0:
                    fp += 1
                if gt_frame[y,x][0] != 0 and gmg_frame[y,x][0] == 0:    
                    fn += 1
                if gt_frame[y,x][0] != 0 and gmg_frame[y,x][0] != 0:
                    tp += 1
                                       
        #print("Frame {}".format(inx))
        print(inx)
        inx += 1
        print("tp: {}, tn: {}, fp: {}, fn: {}".format(tp,tn,fp,fn))
        #if cv2.waitKey(30) & 0xFF == ord('q'):
            #break
    else:
        print("no frame")
        print("tp: {}, tn: {}, fp: {}, fn: {}, acc: {}, prec: {}".format(tp,tn,fp,fn,(tp+tn)/(tp+tn+fp+fn),tp/(tp+fp)))
        break

gt.release()
gmg.release()

#cv2.destroyAllWindows()

accuracy_gmg = 0
accuracy_mog = 0
accuracy_mog2 = 0
accuracy_knn = 0