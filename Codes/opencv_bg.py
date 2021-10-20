# importing libraries
import numpy as np
import cv2

# creating object
fgbg1 = cv2.bgsegm.createBackgroundSubtractorMOG();
fgbg2 = cv2.createBackgroundSubtractorMOG2();
fgbg3 = cv2.bgsegm.createBackgroundSubtractorGMG();

# capture frames from a camera
cap = cv2.VideoCapture('/mnt/disk2/dataset/mot_anno/set0/cam1/raw_video/cam1_2021-07-06,13_00_05.mp4');
#ret, first = cap.read()
# Save the first image as reference
bg = cv2.imread('/home/fatih/phd/DeepPBM/Codes/Result/epoch30_batch8_z2_lr0.01_iyi/000001.jpg')
first_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
first_gray = cv2.GaussianBlur(first_gray, (21, 21), 0)

# Save the first image as reference
#first_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
#first_gray = cv2.GaussianBlur(first_gray, (21, 21), 0)

inx = 1
while cap.isOpened():
	# read frames
	ret, img = cap.read();
		
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0)

	# In each iteration, calculate absolute difference between current frame and reference frame
	difference = cv2.absdiff(gray, first_gray)
	# Apply thresholding to eliminate noise
	thresh = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)[1]
	thresh = cv2.dilate(thresh, None, iterations=2)	
	cv2.imwrite('/home/fatih/phd/DeepPBM/Codes/Result/opencv_bg/deep_opencv/{}.jpg'.format(str(inx).zfill(6)), thresh);	
		
	# apply mask for background subtraction
	#fgmask1 = fgbg1.apply(img);
	#fgmask2 = fgbg2.apply(img);
	#fgmask3 = fgbg3.apply(img);
	
	#cv2.imshow('Original', img);
	#cv2.imwrite('/home/fatih/phd/DeepPBM/Codes/Result/opencv_bg/mog/{}.jpg'.format(str(inx).zfill(6)), fgmask1);
	#cv2.imwrite('/home/fatih/phd/DeepPBM/Codes/Result/opencv_bg/mog2/{}.jpg'.format(str(inx).zfill(6)), fgmask2);
	#cv2.imwrite('/home/fatih/phd/DeepPBM/Codes/Result/opencv_bg/gmg/{}.jpg'.format(str(inx).zfill(6)), fgmask3);
	inx += 1
	k = cv2.waitKey(30) & 0xff;
	if k == 27:
		break;

cap.release();
cv2.destroyAllWindows();

