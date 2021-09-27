import cv2 as cv
import numpy as np
import math
from PIL import Image, ImageChops
from numpy import asarray



"""
imgBG = cv.imread('/home/fatih/fatih_phd/DeepPBM/Codes/Result/BMC2012/Video_002/bg/imageRec000001_l30.jpg')
imgFG = cv.imread('/home/fatih/fatih_phd/DeepPBM/Codes/Result/BMC2012/Video_002/videoFrames/out1.png')
mask = cv.subtract(imgBG,imgFG)
"""

imgBG = cv.imread('/home/fatih/workspace/DeepPBM/Codes/Result/BMC2012/cam1/cam1_002/infer_batch170_epoch600_lr_1e-4_latentd10/imageRec000001_l10.jpg')
imgFG = cv.imread('/home/fatih/workspace/DeepPBM/Codes/Result/BMC2012/cam1/cam1_002/vidFra/out1.png')
mask = cv.subtract(imgFG, imgBG)

gray = cv.cvtColor(cv.cvtColor(mask, cv.COLOR_BGR2RGB), cv.COLOR_BGR2GRAY)
cv.imwrite('/home/fatih/workspace/DeepPBM/Codes/Result/BMC2012/cam1/cam1_002/mask/mask1.jpg', gray)

(T, threshold) = cv.threshold(gray,0,255,cv.THRESH_BINARY | cv.THRESH_OTSU)
thresholdwithblur = cv.medianBlur(threshold, 15,0)
cv.imwrite('/home/fatih/workspace/DeepPBM/Codes/Result/BMC2012/cam1/cam1_002/mask/mask1_binary.jpg', thresholdwithblur)

masked = cv.bitwise_and(imgFG, imgFG, mask=thresholdwithblur)
maskwithblur = cv.medianBlur(masked, 15,0)
cv.imwrite('/home/fatih/workspace/DeepPBM/Codes/Result/BMC2012/cam1/cam1_002/mask/masked1.jpg', maskwithblur)

#mask = ImageChops.subtract(imgFG, imgBG)
"""
maskGray = skimage.color.rgb2gray(maskOrg)
blur = skimage.filters.gaussian(maskGray, sigma=2)
mask = blur < 0.8

viewer = skimage.viewer.ImageViewer(mask)
viewer.view()


mask = mask.convert('L')
maskArr = asarray(mask)
print(maskArr)
maskArr = maskArr > 127
print(maskArr)
mask = Image.fromarray(mask)
mask = mask.convert('RGB')
"""
# th = 0.25*math.exp(1)

# mask = mask > th

# mask.save('/home/fatih/fatih_phd/DeepPBM/Codes/Result/BMC2012/Video_002/mask/mask3.jpg')

#cv.imwrite('/home/fatih/fatih_phd/DeepPBM/Codes/Result/BMC2012/Video_002/mask/mask1.jpg', mask)

