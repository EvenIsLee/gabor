__author__ = 'lonely'
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 20:10:45 2017

@author: lonely
"""
import cv2
import numpy as np
from math import pi
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
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
"""clf = RandomForestClassifier(random_state=9)
RandomForest = clf.fit(classify_mat, label)
k = 0
result = np.zeros([height, width], np.uint)
for i in range(0, 500):
    for j in range(0, 500):
        mat = merge[i:i + 3, j:j + 3]
        matx = get_gabor_feature(mat)
        if clf.predict([matx]) == [0]:
            result[i:i + 3, j:j + 3] = 0
        else:
            result[i:i + 3, j:j + 3] = 255
            k += 1
cv2.imwrite('res.tiff', result)"""