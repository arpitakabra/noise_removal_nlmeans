import cv2
import numpy as np 
from matplotlib import pyplot as plt 
import os
import sys
import scipy
from scipy import signal
from scipy import misc

#the given function is used for extension on the sides in an image. The extension used is 'mirror' in nature
def extend(img, Y, X):

    h,w = np.shape(img)
    extended = np.zeros((h+2*Y, w+2*X)) #the extension is made using the half neighbourhood window size on all the sides
    extended[Y:h+Y, X:w+X] = img
    
    extended[0:Y,X:w+X] = np.flipud(img[0:Y,:])
    extended[Y+h:2*Y+h,X:w+X] = np.flip(img[h-Y:h,:], axis =0)
    extended[Y:h+Y,0:X] = np.flip(img[:,0:X], axis=1)
    extended[Y:h+Y,X+w:2*X+w] = np.flip(img[:,w-X:w], axis =1)
    extended[0:Y,0:X] = np.flip(np.flip(img[0:Y,0:X],axis = 0), axis = 1)
    extended[Y+h:2*Y+h,0:X] = np.flip(np.flip(img[h-Y:h,0:X],axis = 0), axis = 1)
    extended[0:Y,X+w:2*X+w] = np.flip(np.flip(img[0:Y,w-X:w],axis = 0), axis = 1)
    extended[Y+h:2*Y+h,X+w:2*X+w] = np.flip(np.flip(img[h-Y:h,w-X:w],axis = 0), axis = 1)

    return extended




def nlmeans(img, Wsize, h_param):
    
    h,w = np.shape(img)
    hWin = 4 #half neighbourhood window size
    win = hWin*2+1 #actual neighbourhood size
    vec = 2 #half vector size for implementing vector based NLM
    block_size = (2*hWin+1)^2 #block size

    img = extend(img,hWin,hWin) #extended image
    h = h+2*hWin #changed dimensions
    w = w+2*hWin
    # print(h,w)
    denoised = np.zeros((h,w,25)) #output image
    weights = np.zeros((h,w)) 
    variance = np.zeros((h,w)) #local noise variance accumulated here

    #vector based filter is implemented. therefore, each local neighbourhood size is (2*vec+1 x 2*vec+1)
    #the implementation is als based on moving average filter
    for i in range(-Wsize,Wsize+1):
        for j in range(-Wsize,Wsize+1):
            print(". ")
            y_min = max(min(hWin - i,h-hWin),hWin +1) #ranges for x and y to stay inside the image boundaries for computation
            y_max = min(max(h-hWin-1-i, hWin), h-hWin-1)
            x_min = max(min(hWin-j,w-hWin),hWin+1)
            x_max = min(max(w-hWin-1-j, hWin), w-hWin-1)

            if x_min>x_max or y_min>y_max:
                continue
            #range_x and range_y represent the computation window from minimum to maximum of each axis.
            range_y = 1 + np.array(range(y_min,y_max+1))
            range_x = 1 + np.array(range(x_min,x_max+1))
            rx_trans = np.atleast_2d(range_x).T
            sfilter = np.zeros((h,w))
            #an image window is taken and subtracted with just the neighbourhood window, and the resultant is convolved with ones
            temp1 = scipy.signal.convolve2d( np.square(img[range_y-1,rx_trans-1]-img[range_y-1+i,rx_trans-1+j]).T, np.ones((1,2*hWin+1)), mode='same' )
            sfilter[range_y-1, rx_trans-1] = scipy.signal.convolve2d( temp1, np.ones((2*hWin+1,1)), mode='same' ).T
            #wegihts are found by gauss distribution 
            gaussian_weights = (np.exp(-(sfilter[range_y-1, rx_trans-1])/(block_size*h_param*h_param)))
            #the same moving filter approach is used with weights and variance also
            weights[range_y-1, rx_trans-1] = weights[range_y-1, rx_trans-1] + gaussian_weights
            weights[range_y-1+i, rx_trans-1+j] = weights[range_y-1+i, rx_trans-1+j] + gaussian_weights
            variance[range_y-1, rx_trans-1] = variance[range_y-1, rx_trans-1] + np.square(gaussian_weights)
            variance[range_y-1+i, rx_trans-1+j] = variance[range_y-1+i, rx_trans-1+j] + np.square(gaussian_weights)
            
            for k in range(-vec,vec+1): #the denoised image is also calculated by the same moving average filter approach along with combining gaussian weights
                for l in range(-vec, vec+1): #the denoised image is the weighted summation of the shifted windows
                    num = (vec+k)*(2*vec+1)+vec+l+1
                    shifted = np.roll(img, k, axis=0)
                    shifted = np.roll(shifted, l, axis=1)
                    denoised[range_y-1, rx_trans-1,num-1] = ((denoised[range_y-1, rx_trans-1, num-1]) + np.multiply(gaussian_weights,shifted[range_y-1+i, rx_trans-1+j]))
                    denoised[range_y-1+i, rx_trans-1+j,num-1] = ((denoised[range_y-1+i, rx_trans-1+j, num-1]) + np.multiply(gaussian_weights,shifted[range_y-1, rx_trans-1]))


    
    #combining the image under all the vectors  :
    w1 = scipy.signal.convolve2d(weights, np.ones((2*vec+1, 2*vec+1)), mode='same')
    w1[w1<1e-20] = 1e-20 #taking a threshold for weights
    img_final = np.zeros(np.shape(img))

    for k in range(-vec,vec+1):
        for l in range(-vec, vec+1):
            num = (vec+k)*(2*vec+1)+vec+l+1
            shifted = np.roll(denoised[:,:,num], -k, axis=0) #adding the shifted vectors in the final image
            shifted = np.roll(shifted, -l, axis=1) 
            img_final = img_final + np.divide(shifted,w1)

    denoised = img_final
    del img_final

    denoised = denoised[hWin:h-hWin,hWin:w-hWin] #removing the extra padding
    return denoised

            



image_path = "F:\\PRP\\Asgmnt4\\gn_noise\\noise_gn1.jpg"
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
Wsize = 15
sigma = 20
h_param = 2.08*sigma 
denoised = nlmeans(img, Wsize, h_param)

#plotting images:
plt.gray()
fig1=plt.figure(1)
axes=[]
axes.append( fig1.add_subplot(1, 1, 1) ) #plot for estimated ground truth
subplot_title1=("Noisy Image") #noisy image
axes[-1].set_title(subplot_title1)  
plt.imshow(img) 
plt.show()

fig2 = plt.figure(2)
axes.append( fig2.add_subplot(1, 1, 1) )
subplot_title2=("Denoised Image") #denoisy image obtained from nl means
axes[-1].set_title(subplot_title2) 
plt.imshow(denoised)
# fig.tight_layout()    
plt.show()

plt.imsave("denoised1"+".jpg", denoised, cmap = "gray")



