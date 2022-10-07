The given zip folder consists of example images and the following python files:
- addnoise.py: contains functions to add Gaussian noise and salt pepper noise.
               The mean and variance for Gaussian noise, as well as threshold for salt pepper noise can be changed.
- nonlocalmeans.py: contains the definition for computing non local means noise removal algorithm. The paramters that can be changed are Wsize (search window size),
                    and the h_param (parameter for computing weights)
-remove_gn.py: contains function for gaussian noise removal.
	             Uses OpenCV inbuilt Gaussian Smoothening Filter
-mse_psnr.py: Computes MSE and PSNR

In all the files, the path of the root folder and the image should be changed accordingly while running