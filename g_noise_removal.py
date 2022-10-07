import cv2
import numpy as np
import os
from matplotlib import pyplot as plt 

size = 5 #should be an odd number
mean = 0
s = 0.05
gaussian_array = np.zeros((size,size),float)
