__author__ = 'lonely'

from skimage.filters import gabor
from skimage import io
import cv2
from matplotlib import pyplot as plt
from math import pi
import numpy as np
from numpy.random import randint
r = 5
img = cv2.imread('1.tiff',0)
k=0
h,w = img.shape
get_point_h = randint(6,h-10,20)
get_point_w = randint(6,w-10,20)
print get_point_h
print get_point_w
for k in range(20):
    img_seg = img[get_point_h[k]-r:get_point_h[k]+r,get_point_w[k]-r:get_point_w[k]+r]
    for i in range(5):
        for j in range(8):
            filt_real = gabor(img_seg, frequency=(i+1)/10.0,theta=j*pi/8)[0]
            filt_imag = gabor(img_seg, frequency=(i+1)/10.0,theta=j*pi/8)[1]
            res = filt_imag*filt_imag + filt_real*filt_real
            plt.subplot(5,8,k+1)
            k+=1
            plt.imshow(res)
            plt.axis('off')
plt.show()
