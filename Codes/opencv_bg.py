# importing libraries
import os

import numpy as np
import cv2
import time
from natsort import natsorted

# creating object
fgbg1 = cv2.createBackgroundSubtractorMOG2(	history=200,
 											varThreshold=16,
 											detectShadows=True)
fgbg2 = cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames=120,
 													decisionThreshold=0.8)
fgbg3 = cv2.createBackgroundSubtractorKNN(detectShadows=True)

# capture frames from a camera
input_path = '/home/ubuntu/phd/DeepPBM/Data/bmc_real_352x288/Video_008/train_img/'
out_root = "/home/ubuntu/phd/DeepPBM/experiment_bmc/Video_008/"

time_mog2 = 0
time_gmg = 0
time_knn = 0

input_list = os.listdir(input_path)
input_list = natsorted(input_list)

inx = 1
for inp in input_list:
	input_img = cv2.imread(input_path+inp)
	input_img = cv2.GaussianBlur(input_img, (7, 7), 3)

	fgmask1 = fgbg1.apply(input_img)  # mog2
	_, fgmask1 = cv2.threshold(fgmask1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

	fgmask2 = fgbg2.apply(input_img)  # gmg
	_, fgmask2 = cv2.threshold(fgmask2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

	fgmask3 = fgbg3.apply(input_img)  # knn
	_, fgmask3 = cv2.threshold(fgmask3, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

	cv2.imwrite(out_root + 'mog2_mask/{}.jpg'.format(str(inx)), fgmask1)
	cv2.imwrite(out_root + 'gmg_mask/{}.jpg'.format(str(inx)), fgmask2)
	cv2.imwrite(out_root + 'knn_mask/{}.jpg'.format(str(inx)), fgmask3)
	inx += 1


"""
print("Number of frame: {}".format(inx))
print("MOG duration total: {} ;per image: {}".format(time_mog,time_mog/inx))
print("MOG2 duration total: {} ;per image: {}".format(time_mog2,time_mog2/inx))
print("GMG duration total: {} ;per image: {}".format(time_gmg,time_gmg/inx))
print("KNN duration total: {} ;per image: {}".format(time_knn,time_knn/inx))
"""



