import numpy as np
import cv2
import os

#the function takes only gray-scale images as input
def mse_psnr(groundtruth, noisy):
    
    mse = np.mean(np.square(groundtruth-noisy)) #MSE
    psnr = 20*np.log10(255/np.sqrt(mse)) #PSNR 

    return [mse,psnr]
    
root1 = "F:\\PRP\\Asgmnt4\\Data\\Data"
root2 = os.path.abspath('F:\\PRP\\Asgmnt4\\gaussian_filtering')
i=1

for i in range(1,11):
    # path1 = root1+"\\"+"Image"+str(i)+".png"
    gt = cv2.imread("F:\\PRP\\Asgmnt4\\Data\\Data\\Image1.png")
    gt =cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)

    # path2 = root2+"\\"+"noise_gn"+str(i)+".jpg"
    denoised = cv2.imread("F:\\PRP\\Asgmnt4\\nlm_denoised_g1.jpg")
    denoised =cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)

    print(mse_psnr(gt,denoised))
    

