# importing libraries
import numpy as np
import cv2
import time
from matplotlib import pyplot as plt

root_path = "/home/fatih/mnt/datasets/trackeveryseason/cam1/annotations/"

inx = 1

anno_txt = "mot/cam1_2021-07-06,13_00_05.txt"
input_frm = "yolo/cam1_2021-07-06,13_00_05/images"
output = "binary_mask/"

last_frm = False

file = open(root_path+anno_txt, 'r')
lines = file.readlines()
template = np.zeros((1080,1920))
for line in lines:
    line = line.split(',') 
    if inx == int(line[0]):
        cv2.rectangle(template, (int(line[2]), int(line[3])), (int(line[2])+int(line[4]), int(line[3])+int(line[5])),(255,255,255) ,-1)                                                                              
    else:
        str_in = str(inx).rjust(6,'0')
        cv2.imwrite(root_path+output+ str_in + '.jpg',template) 
        template = np.zeros((1080,1920))  
        cv2.rectangle(template, (int(line[2]), int(line[3])), (int(line[2])+int(line[4]), int(line[3])+int(line[5])),(255,255,255) ,-1)     
        inx += 1  
    if inx == 1800: last_frm = True
    if last_frm:
        str_in = str(inx).rjust(6,'0')
        cv2.imwrite(root_path+output+ str_in + '.jpg',template)       
            