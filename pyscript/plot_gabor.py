__author__ = 'lee'
import cv2
import numpy as np
from math import pi
from skimage.filters import gabor

merge = cv2.imread('merge.tiff', 0)
height, width = merge.shape
img1 = cv2.imread('1.tiff', 0)
img2 = cv2.imread('2.tiff', 0)

train = 100

random1 = np.random.randint(3, 512, train)
random2 = np.random.randint(3, 512, train)

classify_mat = np.zeros([200, 40], np.float32)
k = 0
label = []


def get_gabor_feature(image):
    classify = np.array([])
    for i in range(0, 5, 1):
        for j in range(0, 8, 1):
            filt_real = gabor(image, frequency=(i+1)/10.0, theta=j*pi/8)[0]
            filt_imag = gabor(image, frequency=(i+1)/10.0, theta=j*pi/8)[1]
            res = filt_real * filt_real + filt_imag * filt_imag
            res_mean = np.mean(res)
            classify = np.append(classify,res_mean)
    return classify


for i in range(train):
    mat = img1[random1[i]-3:random1[i] + 3, random2[i]-3:random2[i] + 3]
    classify_mat[k] = get_gabor_feature(mat)
    k += 1
    label.append(0)
for i in range(train):
    mat = img2[random1[i]-3:random1[i] + 3, random2[i]-3:random2[i] + 3]
    classify_mat[k] = get_gabor_feature(mat)
    print classify_mat[k]
    k += 1
    label.append(1)


np.savetxt('train.txt',classify_mat)
np.savetxt('label.txt',label)
