import cv2 as cv
import numpy as np
import math
from PIL import Image, ImageChops
from numpy import asarray
import os
import copy


path = "/home/fatih/phd/CenterNet_DeepPBM/set0/exp/gt/annotations/"

files = os.listdir(path)

for index, file in enumerate(files):
    file_new = file.split('.')[0]
    file_new = int(file_new) +1
    file_new = str(file_new).rjust(6,'0')
    file_new = file_new + '.txt'
    os.rename(os.path.join(path, file), os.path.join('/home/fatih/phd/CenterNet_DeepPBM/set0/exp/gt/cnv/', file_new))