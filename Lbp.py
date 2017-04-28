__author__ = 'lonely'
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 15:23:18 2017

@author: lonely
"""

import numpy as np
import cv2
from sklearn import svm
from skimage.feature import local_binary_pattern
from sklearn import preprocessing

t1 = cv2.imread('merge.tiff',0)

height = t1.shape[0]
width = t1.shape[1]
reflect = 5
t = cv2.copyMakeBorder(t1, reflect, reflect, reflect, reflect, cv2.BORDER_REFLECT)
c1 = cv2.imread('1.tiff',0)

c2 = cv2.imread('2.tiff',0)

train = 200
classify_mat = np.zeros([2*train, 255], np.float32)
label = []
k = 0
a = 20
random1 = np.random.randint(0, 512, train)
random2 = np.random.randint(0, 512, train)
for i in range(train):
    mat = c1[random1[i]:random1[i] + a, random2[i]:random2[i] + a]
    classify_mat[k] = np.histogram(local_binary_pattern(mat, 8, 1), bins=255)[0]
    k += 1
    label.append(0)

for i in range(train):
    mat = c2[random1[i]:random1[i] + a, random2[i]:random2[i] + a]
    classify_mat[k] = np.histogram(local_binary_pattern(mat, 8, 1), bins=255)[0]
    k += 1
    label.append(1)


#KNN = neighbors.KNeighborsClassifier()
#KNN.fit(classify_mat, label)

C = 1.0
svc = svm.SVC(kernel='linear', C=C).fit(classify_mat, label)
k=0
result = np.zeros([height, width], np.uint)
for i in range(reflect, height+reflect):
    for j in range(reflect, width+reflect):
        mat = t[i-reflect:i+reflect, j-reflect:j+reflect]
        mat = np.histogram(local_binary_pattern(mat, 8, 1), bins=255)[0]
        if svc.predict([mat]) ==[0]:
            result[i-reflect, j-reflect] = 0
        else:
            result[i-reflect, j-reflect] = 255
        k += 1
        print k
cv2.imwrite('141045522.tiff', result)