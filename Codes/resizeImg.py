import cv2
import os

input_dir = "/home/ubuntu/phd/dataset/bmc2012/bmc_real/"
output_dir = "/home/ubuntu/phd/dataset/bmc2012/bmc_real_352x288/"
videoNum = 8
folderName = "eval_img"

input_path = input_dir + "Video_00{}/{}/".format(videoNum, folderName)
output_path = output_dir + "Video_00{}/{}/".format(videoNum, folderName)

images = os.listdir(input_path)
for img in images:
    im1 = cv2.imread(input_path + img)
    im1 = cv2.resize(im1, (352, 288), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(output_path + img.split('.')[0] + ".jpg", im1)
