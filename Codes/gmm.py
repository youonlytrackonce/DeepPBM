# importing libraries
import numpy as np
import cv2
import time

# creating object
fgbg1 = cv2.bgsegm.createBackgroundSubtractorMOG()
fgbg2 = cv2.createBackgroundSubtractorMOG2()
fgbg3 = cv2.bgsegm.createBackgroundSubtractorGMG()
fgbg4 = cv2.createBackgroundSubtractorKNN(detectShadows=False)

out_root = "/home/fatih/phd/experiments/inference/bg_model/gmmBGmodel/koray/"

# capture frames from a camera
cap = cv2.VideoCapture('/mnt/disk2/dataset/trackeveryseason/cam1/raw_video/cam1_2021-07-06,13_00_05.mp4')

time_mog = 0

inx = 1
while cap.isOpened():
	# read frames
    ret, img = cap.read()
    if ret:	
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        img1 = gray.copy() 

        start1 = time.time()
        fgmask1 = fgbg3.apply(img1)
        time_mog += time.time() - start1
       
        cv2.imwrite(out_root + 'gmg/{}.jpg'.format(str(inx).zfill(6)), fgmask1)
        kernel = np.ones((7,7),np.uint8)
        opening = cv2.morphologyEx(fgmask1, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, (7, 7))

        cv2.imwrite(out_root + 'gmg_openclose/{}.jpg'.format(str(inx).zfill(6)), closing)
        inx += 1

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    else:
        cap.release()
        cv2.destroyAllWindows()
        inx -= 1	

print("Number of frame: {}".format(inx))
print("MOG2 duration total: {} ;per image: {}".format(time_mog,time_mog/inx))