# importing libraries
import numpy as np
import cv2
import time
from matplotlib import pyplot as plt

root_path = "/home/fatih/phd/experiments/inference/bg_model/gmmBGmodel/"

inx = "001000"

raw = cv2.imread(root_path + "raw/{}.jpg".format(inx))
mog = cv2.imread(root_path + "mog/{}.jpg".format(inx))
mog2 = cv2.imread(root_path + "mog2/{}.jpg".format(inx))
gmg = cv2.imread(root_path + "gmg/{}.jpg".format(inx))
knn = cv2.imread(root_path + "knn/{}.jpg".format(inx))
dpbm = cv2.imread(root_path + "deeppbm_mask/{}.jpg".format(inx))

# create figure
fig = plt.figure(figsize=(300, 300))
# setting values to rows and column variables
rows = 3
columns = 2

# Adds a subplot at the 1st position
fig.add_subplot(rows, columns, 1)
  
# showing image
plt.imshow(raw)
plt.axis('off')
plt.title("INPUT")
  
# Adds a subplot at the 2nd position
fig.add_subplot(rows, columns, 2)
  
# showing image
plt.imshow(mog)
plt.axis('off')
plt.title("MOG")
  
# Adds a subplot at the 3rd position
fig.add_subplot(rows, columns, 3)
  
# showing image
plt.imshow(mog2)
plt.axis('off')
plt.title("MOG2")
  
# Adds a subplot at the 4th position
fig.add_subplot(rows, columns, 4)
  
# showing image
plt.imshow(gmg)
plt.axis('off')
plt.title("GMG")

# Adds a subplot at the 5th position
fig.add_subplot(rows, columns, 5)

# showing image
plt.imshow(knn)
plt.axis('off')
plt.title("KNN")

# Adds a subplot at the 6th position
fig.add_subplot(rows, columns, 6)

# showing image
plt.imshow(dpbm)
plt.axis('off')
plt.title("DeepPBM")


plt.show()


"""
cv2.imshow("RAW", raw)
cv2.waitKey()

cv2.imshow("MOG", mog)
cv2.waitKey()

cv2.imshow("MOG2", mog2)
cv2.waitKey()

cv2.imshow("GMG", gmg)
cv2.waitKey()

cv2.imshow("KNN", knn)
cv2.waitKey()
"""