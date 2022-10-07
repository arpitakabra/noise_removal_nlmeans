import cv2
import numpy as np
import os
import sys
from PIL import Image
from matplotlib import pyplot as plt 
import random

def add_gn(path,mean,variance):
    img=cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img/255
    noise =  np.random.normal(mean,variance,np.shape(img))
    noise = noise.reshape(np.shape(img))
    noised = img+noise
    plt.imsave("noise_gn"+str(i)+".jpg", noised, cmap = "gray")

def add_spn(path,salt,pepper):
    img=cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h,w = np.shape(img)
    # img = img/255
    sp_noise = np.random.randint(0,256,(h,w))
    img = np.where(sp_noise<=pepper,0,img)
    img = np.where(sp_noise>=salt,255,img)
    plt.imsave("noise_spn"+str(i)+".jpg", img, cmap = "gray")
    

root = os.path.abspath("F:\\PRP\\Asgmnt4\\Data\\Data")
i=1

mean = 0 #mean for adding noise
s = 0.05 #standard deviation of added noise

pepper = 4
salt = 255-pepper

for data in sorted(os.listdir(root)):
    path = root+"\\"+str(data)
    add_gn(path,mean,s)
    add_spn(path,salt,pepper)
    i+=1