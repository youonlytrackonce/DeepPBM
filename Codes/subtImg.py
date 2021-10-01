import cv2 as cv
import numpy as np
import math
from PIL import Image, ImageChops
from numpy import asarray
import os



"""
imgBG = cv.imread('/home/fatih/fatih_phd/DeepPBM/Codes/Result/BMC2012/Video_002/bg/imageRec000001_l30.jpg')
imgFG = cv.imread('/home/fatih/fatih_phd/DeepPBM/Codes/Result/BMC2012/Video_002/videoFrames/out1.png')
mask = cv.subtract(imgBG,imgFG)
"""

root_path = "/home/fatih/phd/CenterNet_DeepPBM/set0/cam1/"

imgBG = cv.imread(root_path+'bg/000001.jpg')
imgFG = cv.imread(root_path+'fg/000001.jpg')

mask = cv.subtract(imgBG,imgFG)
mask2 = cv.absdiff(imgBG,imgFG)
cv.imwrite(root_path+'mask/absdiff_000000.jpg', mask)

Conv_hsv_Gray = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
cv.imwrite(root_path+'mask/absdiffgray_000000.jpg', Conv_hsv_Gray)

gaussBlur = cv.GaussianBlur(Conv_hsv_Gray, (5, 5), 0)
cv.imwrite(root_path+'mask/gaussblur_000000.jpg', gaussBlur)

medianBlur = cv.medianBlur(Conv_hsv_Gray, 5,0)
cv.imwrite(root_path+'mask/medianblur_000000.jpg', medianBlur)

averageblur = cv.blur(Conv_hsv_Gray, (5,5))
cv.imwrite(root_path+'mask/averageblur_000000.jpg', averageblur)

(gT, gaussblurotsu) = cv.threshold(gaussBlur,0,255,cv.THRESH_BINARY + cv.THRESH_OTSU)
cv.imwrite(root_path+'mask/gaussblurotsu_000000.jpg', gaussblurotsu)

(mT, medianblurotsu) = cv.threshold(medianBlur,0,255,cv.THRESH_BINARY + cv.THRESH_OTSU)
cv.imwrite(root_path+'mask/medianblurotsu_000000.jpg', medianblurotsu)

(mT, averageblurotsu) = cv.threshold(averageblur,0,255,cv.THRESH_BINARY + cv.THRESH_OTSU)
cv.imwrite(root_path+'mask/averageblurotsu_000000.jpg', averageblurotsu)




"""
cv.imwrite(root_path+'mask/otsu_000000.jpg', abs_mask)
cv.imwrite(root_path+'mask/blur_000000.jpg', thresholdwithblur)
cv.imwrite(root_path+'mask/mask2_000000.jpg', mask2)
cv.imwrite(root_path+'mask/imgBG_000000.jpg', imgBG)
cv.imwrite(root_path+'mask/imgFG_000000.jpg', imgFG)
##############################################

mask2[abs_mask != 255] = [0,0,0]
imgBG[abs_mask != 255] = [0,0,0]
imgFG[abs_mask != 255] = [0,0,0] 
"""


# mask3 = imgFG - imgBG
#gray = cv.cvtColor(cv.cvtColor(mask, cv.COLOR_BGR2RGB), cv.COLOR_BGR2GRAY)

# cv.imwrite(root_path+'mask/negate_color_000000.jpg', mask3)

# gray = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
# cv.imwrite(root_path+'mask/subt_000000.jpg', gray)
# cv.imwrite(root_path+'mask/mask_000000.jpg', mask)


"""
(T, threshold) = cv.threshold(gray,0,255,cv.THRESH_BINARY | cv.THRESH_OTSU)
thresholdwithblur = cv.medianBlur(threshold, 15,0)
cv.imwrite(root_path+'mask/binary_otsu_000000.jpg', thresholdwithblur)

masked = cv.bitwise_and(imgFG, imgFG, mask=thresholdwithblur)
maskwithblur = cv.medianBlur(masked, 15,0)
cv.imwrite(root_path+'mask/masked_000000.jpg', maskwithblur)

#mask = ImageChops.subtract(imgFG, imgBG)

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

