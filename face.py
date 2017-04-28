__author__ = 'lonely'
from sklearn.datasets import fetch_lfw_people

lfw_people = fetch_lfw_people(min_faces_per_person=1 ,resize=0.4)
n_samples, h, w = lfw_people.images.shape
print n_samples,h,w