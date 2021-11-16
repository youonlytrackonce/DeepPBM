import cv2 as cv
import numpy as np
import math
from PIL import Image, ImageChops
from numpy import asarray
import os
import copy
import time



"""
imgBG = cv.imread('/home/fatih/fatih_phd/DeepPBM/Codes/Result/BMC2012/Video_002/bg/imageRec000001_l30.jpg')
imgFG = cv.imread('/home/fatih/fatih_phd/DeepPBM/Codes/Result/BMC2012/Video_002/videoFrames/out1.png')
mask = cv.subtract(imgBG,imgFG)
"""

root_path = "/home/fatih/phd/CenterNet_DeepPBM/set0/cam1/"

files = os.listdir(root_path+'bg')

start = time.time()
for refImg in files:

    imgBG = cv.imread(root_path+'bg/' + refImg)
    imgFG = cv.imread(root_path+'fg/' + refImg)

    imgFG = cv.GaussianBlur(imgFG, (7,7), 3)

    # mask = cv.subtract(imgBG,imgFG)
    diffImg = cv.absdiff(imgBG,imgFG)
    #cv.imwrite(root_path+'mask/absdiff_000001.jpg', diffImg)

    Conv_hsv_Gray = cv.cvtColor(diffImg, cv.COLOR_BGR2GRAY)
    #cv.imwrite(root_path+'diff/'+refImg, Conv_hsv_Gray)

    blurred = cv.GaussianBlur(Conv_hsv_Gray, (7,7), 3)
    #cv.imwrite(root_path+'mask/absdiffgrayblur_000001.jpg', blurred)

    #thresh = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 21, 10)
    #cv.imwrite(root_path+'mask/adaptive_000001.jpg', thresh)

    #mask1 = copy.deepcopy(blurred)
    mask2 = copy.deepcopy(blurred)
    """
    (gT, threshotsu) = cv.threshold(mask1,0,255,cv.THRESH_OTSU)
    cv.imwrite(root_path+'mask/threshotsu_000001.jpg', threshotsu)
    gaussBlurOtsu = cv.GaussianBlur(threshotsu, (7, 7), 0)
    cv.imwrite(root_path+'mask/gaussblurOtsu_000001.jpg', gaussBlurOtsu)
    """

    (gT, threshtri) = cv.threshold(mask2,0,255,cv.THRESH_TRIANGLE)
    #cv.imwrite(root_path+'mask/threshtri_000001.jpg', threshtri)
    gaussBlurTri = cv.GaussianBlur(threshtri, (7, 7), 0)
    #cv.imwrite(root_path+'mask/' + refImg, gaussBlurTri)

    normMask = gaussBlurTri/255

    res = cv.bitwise_and(imgFG,imgFG,mask = gaussBlurTri)
    #cv.imwrite(root_path+'masked/' + refImg, res)
    """
    mask2 = copy.deepcopy(Conv_hsv_Gray)
    mask3 = copy.deepcopy(Conv_hsv_Gray)

    ###########################################################
    #gaussBlur = cv.GaussianBlur(mask1, (5, 5), 0)
    #cv.imwrite(root_path+'mask/gaussblur_000001.jpg', gaussBlur)

    (gT, gaussblurotsu) = cv.threshold(mask1,0,255,cv.THRESH_BINARY + cv.THRESH_OTSU)
    cv.imwrite(root_path+'mask/gaussblurotsu_000001.jpg', gaussblurotsu)
    ###########################################################
    #medianBlur = cv.medianBlur(mask2, 5,0)
    #cv.imwrite(root_path+'mask/medianblur_000001.jpg', medianBlur)

    (mT, medianblurotsu) = cv.threshold(mask2,0,255,cv.THRESH_BINARY + cv.THRESH_OTSU)
    cv.imwrite(root_path+'mask/medianblurotsu_000001.jpg', medianblurotsu)

    ###########################################################
    averageblur = cv.blur(mask3, (5,5))
    cv.imwrite(root_path+'mask/averageblur_000001.jpg', averageblur)

    (mT, averageblurotsu) = cv.threshold(averageblur,0,255,cv.THRESH_BINARY + cv.THRESH_OTSU)
    cv.imwrite(root_path+'mask/averageblurotsu_000001.jpg', averageblurotsu)
    """



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

total = time.time() - start
print(total)