import numpy as np 
import cv2
from matplotlib import pyplot as plt 
import os

root = os.path.abspath("F:\\PRP\\Asgmnt4\\sp_noise") #noisy images containing folder

i=1
for data in sorted(os.listdir(root)):
    path = root+"\\"+str(data)
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img,(11,11),0.85) #applying Gaussian filter which is an inbuilt command in cv2
    plt.imsave("gaussian_filtering\\noise_sp"+str(i)+".jpg", blur, cmap = "gray") 
    i=i+1
