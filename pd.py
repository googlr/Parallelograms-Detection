# -*- coding: utf-8 -*-

# The program will consist of three steps: 
#	(1) detect edges using the Sobel’s operator, 
#	(2) detect straight line segments using the Hough Transform, and 
#	(3) detect parallelograms from the straight-line segments detected in step (2). 
# In step (1), compute edge magnitude using the formula below and 
# then normalize the magnitude values to lie within the range [0,255]. 
# Next, manually choose a threshold value to produce a binary edge map.

import numpy as np
from matplotlib import pylab as plt
import math


# convert them into grayscale images by using the formula 
# luminance = 0.30R + 0.59G + 0.11B, 
# where R, G, and B, are the red, green, and blue components.
def cvt2grayscale(img):
	grayImage = []
	for i in range(0, img.size/3):
		luminance = int(0.3*img[3*i] + 0.59*img[3*i + 1] + 0.11*img[3*i + 2])
		grayImage.append(luminance)

	return np.array(grayImage)

def sobels_operator(img):
	mag = []
	img_row, img_col = img.shape
	for i in range(1, img_row  - 1):
		for j in range(1, img_col - 1):
			g_x = (img[i-1][j+1] + 2*img[i][j+1] + img[i+1][j+1] 
				- img[i-1][j-1] - 2*img[i][j-1] - img[i+1][j-1])
			g_y = (img[i-1][j-1] + 2*img[i-1][j] + img[i-1][j+1] 
				- img[i+1][j-1] - 2*img[i+1][j] - img[i+1][j+1])
			mag_i_j = math.sqrt(g_x*g_x + g_y*g_y)
			# T=225
			mag_i_j = mag_i_j if mag_i_j >= 225 else 0
			#mag_i_j = mag_i_j if mag_i_j <= 225 else 255
			mag.append(mag_i_j)

	return np.array(mag).reshape([img_row-2, img_col-2])


row, col = 756, 1008
filename = "TestImage2.raw"
#Read Image
testImage = np.fromfile(filename,dtype='uint8',sep="")

# Convert to grayscale image
grayImage = cvt2grayscale(testImage).reshape([row, col])
print("Step 1: Convert image to grayscale.")
#print grayImage.shape

#Display Image
#plt.imshow(grayImage)
#plt.show()

#################################################################
#(1) detect edges using the Sobel’s operator
#– Filtering
#– Enhancement
imgMag = sobels_operator(grayImage)
print("Step 2: Sobel's operator applied.")
#plt.imshow(imgMag)
#plt.show()

#################################################################
#(2) detect straight line segments using the Hough Transform
theta_step_size = 1
p_step_size = 1
theta_MAX_VALUE = 360
p_MAX_VALUE = int( math.sqrt(row*row + col*col) )
accumulator_array = np.zeros((theta_MAX_VALUE/theta_step_size, p_MAX_VALUE),dtype='uint8')
#Compute the accumulator array
imgMag_row, imgMag_col = imgMag.shape
for i in range(0, imgMag_row):
	for j in range(0, imgMag_col):
		if( imgMag[i][j] == 0):
			continue
		# p = x*cos(theta) + y*sin(theta)
		theta = 0
		while theta < 360:
			theta_radians = math.radians(theta + theta_step_size/2.0)
			p_estimate = i*math.cos(theta_radians) + j*math.sin(theta_radians)
			#Update the accumulator array
			accu_x = theta/theta_step_size
			accu_y = int( p_estimate )
			accumulator_array[ accu_x ][ accu_y ] += 1
			# next theta
			theta = theta + theta_step_size
print(accumulator_array.shape)
print( np.amax(accumulator_array) )
plt.imshow(accumulator_array, cmap='gray')
plt.show()


#Saving filtered image to new file
