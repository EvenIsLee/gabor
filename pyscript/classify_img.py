__author__ = 'lee'
import numpy as np
import cv2
from skimage.filters import gabor
from math import pi
from sklearn import svm


img = cv2.imread('merge.tiff', 0)
height, width = img.shape


def get_gabor_feature(image):
    classify = np.array([])
    for i in range(0, 5, 1):
        for j in range(0, 8, 1):
            filt_real = gabor(image, frequency=(i+1)/10.0, theta=j*pi/8)[0]
            filt_imag = gabor(image, frequency=(i+1)/10.0, theta=j*pi/8)[1]
            res = filt_real * filt_real + filt_imag * filt_imag
            res_mean = np.mean(res)
            classify = np.append(classify,res_mean)
    return np.array(classify)


train = np.loadtxt('train.txt')
label = np.loadtxt('label.txt')

clf = svm.SVC(C=1.0, kernel='linear')
clf.fit(train, label)

result = np.zeros([height, width], np.uint)
for i in range(3, 500):
    for j in range(3, 500):
        area_mat = img[i-3:i+3, j-3:j+3]
        area_fea = get_gabor_feature(area_mat)
        if clf.predict([area_fea]) == [0]:
            result[i-3:i+3, j:j+3] = 0
        else:
            result[i-3:i+3, j:j+3] = 255

cv2.imwrite('res.tiff', result)