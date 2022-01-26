import cv2 as cv
import matplotlib.pyplot as plt

videos = [2, 3, 4, 6, 7, 8]

vid_path = '/home/ubuntu/Documents/bmc2012/video'

gt = 'Mask.jpg'
img = 'Img.jpg'
gmg = 'gmg.jpg'
knn = 'knn.jpg'
mog2 = 'mog2.jpg'
resnet = 'resnet.jpg'
vanilla = 'vanilla.jpg'

images = [img, gt, resnet, vanilla, knn, gmg, mog2]
f, axarr = plt.subplots(7, 6, figsize=(10, 10))
for y, vi in enumerate(videos):
    vid = vid_path + str(vi) + '/'
    for x, im in enumerate(images):
        im_name = im.split('.')[0]
        axarr[x, y].imshow(cv.imread(vid + im))
        axarr[x, y].set_title(im_name, fontsize=5)
        axarr[x, y].set_axis_off()
plt.savefig('/home/ubuntu/Documents/bmc2012/fig.png')
plt.show()
plt.close()
