# importing libraries
import numpy as np
import cv2
import time

# creating object
fgbg1 = cv2.bgsegm.createBackgroundSubtractorMOG();
fgbg2 = cv2.createBackgroundSubtractorMOG2();
fgbg3 = cv2.bgsegm.createBackgroundSubtractorGMG();
fgbg4 = cv2.createBackgroundSubtractorKNN(detectShadows=False)

# capture frames from a camera
cap = cv2.VideoCapture('/mnt/disk2/dataset/trackeveryseason/cam1/raw_video/cam1_2021-07-06,13_00_05.mp4');
#ret, first = cap.read()
# Save the first image as reference
#bg = cv2.imread('/home/fatih/phd/DeepPBM/Codes/Result/epoch30_batch8_z2_lr0.01_iyi/000001.jpg')
#first_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
#first_gray = cv2.GaussianBlur(first_gray, (21, 21), 0)

# Save the first image as reference
#first_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
#first_gray = cv2.GaussianBlur(first_gray, (21, 21), 0)

out_root = "/home/fatih/phd/experiments/inference/bg_model/gmmBGmodel/"

time_mog = 0
time_mog2 = 0
time_gmg = 0
time_knn = 0

inx = 1
while cap.isOpened():
	# read frames
	ret, img = cap.read();
	if ret:	
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (21, 21), 0)

		img1 = gray.copy() 
		img2 = gray.copy()
		img3 = gray.copy()
		img4 = gray.copy()
		# In each iteration, calculate absolute difference between current frame and reference frame
		#difference = cv2.absdiff(gray, first_gray)
		# Apply thresholding to eliminate noise
		#thresh = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)[1]
		#thresh = cv2.dilate(thresh, None, iterations=2)	
		#cv2.imwrite('/home/fatih/phd/DeepPBM/Codes/Result/opencv_bg/deep_opencv/{}.jpg'.format(str(inx).zfill(6)), thresh);	
			
		# apply mask for background subtraction
		start1 = time.time()
		fgmask1 = fgbg1.apply(img1)
		time_mog += time.time() - start1

		start2 = time.time()
		fgmask2 = fgbg2.apply(img2)
		time_mog2 += time.time() - start2

		start3 = time.time()
		fgmask3 = fgbg3.apply(img3)
		time_gmg += time.time() - start3

		start4 = time.time()
		fgmask4 = fgbg4.apply(img4)
		time_knn += time.time() - start4

		print(inx)
		#fgmask1 = cv2.GaussianBlur(fgmask1, (21, 21), 0)
		#fgmask2 = cv2.GaussianBlur(fgmask2, (21, 21), 0)
		#fgmask3 = cv2.GaussianBlur(fgmask3, (21, 21), 0)
		#fgmask4 = cv2.GaussianBlur(fgmask4, (21, 21), 0)
		inx += 1
		"""
		fgmask1 = cv2.threshold(fgmask1, 25, 255, cv2.THRESH_BINARY)
		fgmask2 = cv2.threshold(fgmask2, 25, 255, cv2.THRESH_BINARY)
		fgmask3 = cv2.threshold(fgmask3, 25, 255, cv2.THRESH_BINARY)
		fgmask4 = cv2.threshold(fgmask4, 25, 255, cv2.THRESH_BINARY)
		

		#cv2.imshow('Original', img);
		cv2.imwrite(out_root + 'mog/{}.jpg'.format(str(inx).zfill(6)), fgmask1)
		cv2.imwrite(out_root + 'mog2/{}.jpg'.format(str(inx).zfill(6)), fgmask2)
		cv2.imwrite(out_root + 'gmg/{}.jpg'.format(str(inx).zfill(6)), fgmask3)
		cv2.imwrite(out_root + 'knn/{}.jpg'.format(str(inx).zfill(6)), fgmask4)
		inx += 1
		k = cv2.waitKey(30) & 0xff;
		if k == 27:
			break;
		"""
	else:
		cap.release();
		cv2.destroyAllWindows()
		inx -= 1	
print("Number of frame: {}".format(inx))
print("MOG duration total: {} ;per image: {}".format(time_mog,time_mog/inx))
print("MOG2 duration total: {} ;per image: {}".format(time_mog2,time_mog2/inx))
print("GMG duration total: {} ;per image: {}".format(time_gmg,time_gmg/inx))
print("KNN duration total: {} ;per image: {}".format(time_knn,time_knn/inx))



