__author__ = 'lonely'
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from skimage.filters import gabor
from math import pi
from sklearn import svm
merge = cv2.imread('merge.tiff', 0)
height, width = merge.shape


def get_gabor_feature(image):
    classify =np.array([])
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

clf = svm.SVC(C=1.0,kernel='linear').fit(train, label)

k = 0
result = np.zeros([height, width], np.uint)
a = []
for i in range(3, 500):
    for j in range(3, 500):
        mat = merge[i - 3 : i + 3, j - 3 : j + 3]
        matx = get_gabor_feature(mat)
        if clf.predict([matx]) == [0]:
            result[i-3:i + 3, j:j + 3] = 0
        else:
            result[i-3:i + 3, j:j + 3] = 255
        k += 1
        print k
cv2.imwrite('res.tiff', result)