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
import itertools as it


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
filename = "TestImage1.raw"
#Read Image
testImage = np.fromfile(filename,dtype='uint8',sep="")

# Convert to grayscale image
grayImage = cvt2grayscale(testImage).reshape([row, col])
print("Step 1: Convert image to grayscale.")
#print grayImage.shape

#Display Image
#plt.imshow(grayImage, cmap = 'gray')
#plt.show()

#################################################################
#(1) detect edges using the Sobel’s operator
#– Filtering
#– Enhancement
imgMag = sobels_operator(grayImage)
print("Step 2: Sobel's operator applied.")
#plt.imshow(imgMag, cmap = 'gray')
#plt.show()

#################################################################
#(2) detect straight line segments using the Hough Transform
theta_step_size = 5
p_step_size = 1
theta_MAX_VALUE = 360
p_MAX_VALUE = int( math.sqrt(row*row + col*col) )
accumulator_array = np.zeros((theta_MAX_VALUE/theta_step_size, p_MAX_VALUE/p_step_size),dtype='uint8')
#Compute the accumulator array
imgMag_row, imgMag_col = imgMag.shape
for i in range(0, imgMag_row):
	for j in range(0, imgMag_col):
		if( imgMag[i][j] > 0):
			# p = x*cos(theta) + y*sin(theta)
			theta = 0
			while theta < 360:
				theta_radians = math.radians(theta + theta_step_size/2.0)
				p_estimate = i*math.cos(theta_radians) + j*math.sin(theta_radians)
				#Update the accumulator array
				accu_x = theta/theta_step_size
				accu_y = int( p_estimate/p_step_size )
				accumulator_array[ accu_x ][ accu_y ] += 1
				# next theta
				theta = theta + theta_step_size

max_accumulator = np.amax(accumulator_array)
print( max_accumulator )
print( "Step 3: Hough Transform applied.")
#plt.imshow(accumulator_array, cmap='gray')
#plt.show()


#################################################################
#(3) detect parallelograms from the straight-line segments detected in step (2).
#the de-Houghed image (using a relative threshold of 40%)
relative_threshold_ratio = 0.4
relative_threshold = max_accumulator * relative_threshold_ratio
accu_row, accu_col = accumulator_array.shape
peak_list = []
for i in range(0, accu_row):
	for j in range(0, accu_col):
		#apply the threshold filter
		accumulator_i_j = accumulator_array[i][j]
		accumulator_array[i][j] = accumulator_i_j if accumulator_i_j >= relative_threshold else 0
		if accumulator_i_j >= relative_threshold:
			peak_p = (j + 0.5) * p_step_size
			peak_theta = (i + 0.5) * theta_step_size
			peak_list.append([peak_theta, peak_p])


# using local-maxima threshold
#accu_row, accu_col = accumulator_array.shape
#peak_list = []
#for i in range(1, accu_row - 1):
#	for j in range(1, accu_col -1):
#		#apply the threshold filter
#		if (accumulator_array[i][j] >= accumulator_array[i-1][j] and accumulator_array[i][j] >= accumulator_array[i+1][j] and accumulator_array[i][j] >= accumulator_array[i][j-1] and accumulator_array[i][j] >= accumulator_array[i][j+1]):
#			peak_p = (j + 0.5) * p_step_size
#			peak_theta = (i + 0.5) * theta_step_size
#			peak_list.append([peak_theta, peak_p])
#


#################################################################################################
# Filter overlaping lines
filter_step_size = 5

# Compute average of a list of int
def average_p( p_filter_list ):
	list_len = len( p_filter_list )
	if list_len == 0:
		print("Warning: empty list.")
	p_sum = 0.0
	for p in p_filter_list:
		p_sum  = p_sum + p

	return p_sum/list_len

# Cluster a list of int to clustered list
def cluster_list( p_list ):
	p_list = sorted( p_list )
	list_len = len( p_list )
	clustered_list = []
	if list_len == 0:
		return clustered_list
	p_val = p_list[0]
	p_filter_list = []
	for i in range(0, list_len):
		if math.fabs( p_val - p_list[i] ) < filter_step_size:
			p_filter_list.append( p_list[i] )
		else:
			p_new_average = average_p( p_filter_list )
			clustered_list.append( p_new_average )
			# update p_val and clear p_filter_list
			p_val = p_list[i]
			p_filter_list[:] = []
			p_filter_list.append( p_list[i] )

	# clear p_filter_list
	if len( p_filter_list ) != 0:
		p_new_average = average_p( p_filter_list )
		clustered_list.append( p_new_average )
	return clustered_list


#peak_list_filtered = []
#filter_theta = peak_list[0][0]
#filter_p_list = []
#peak_list_len = len( peak_list )
#for i in range(0, peak_list_len ):
#	i_theta = peak_list[i][0]
#	i_p = peak_list[i][1]
#	if i_theta == filter_theta:
#		filter_p_list.append(i_p)
#		continue
#	else:
#		cluster_p_list = cluster_list( filter_p_list )
#		for p in cluster_p_list:
#			peak_list_filtered.append( [ filter_theta, p ] )
		#update filter_theta and clear filter_p_list
#		filter_theta = i_theta
#		filter_p_list[:] = []
#		filter_p_list.append( i_p )
#
	# clear filter_p_list
#	if len( filter_p_list ) != 0 :
#		cluster_p_list = cluster_list( filter_p_list )
#		for p in cluster_p_list:
#			peak_list_filtered.append( [ filter_theta, p ] )


# use dictionary to filter peaks
peak_dict = {}
for line in peak_list:
    if line[0] in peak_dict:
        # append the new number to the existing array at this slot
        peak_dict[line[0]].append(line[1])
    else:
        # create a new array in this slot
        peak_dict[line[0]] = [line[1]]

for key in peak_dict:
	peak_dict[ key ] = cluster_list( peak_dict[key] )

peak_list_filtered = []
for key in peak_dict:
	for val in peak_dict[ key ]:
		peak_list_filtered.append( [ key, val ] )

print( peak_list_filtered )
peak = np.array( peak_list_filtered )

########################################################################################
#print(peak)
edge_map = np.zeros( (row, col), dtype='uint8')
#Initialize to edge map to 255
for i in range(0, row):
	for j in range(0, col):
		edge_map[i][j] = 255

#Copy the magnitude array imgMag to edge_map
for i in range(0, row-2):
	for j in range(0, col-2):
		if imgMag[i][j] > 0:
			edge_map[i+1][j+1] = 0

def xy_in_range(x,y):
	return True if ( x >= 0 and x < row and y >=0 and y < col ) else False
#Draw the lines in edge_map
peak_row, peak_col = peak.shape
for i in range(0, peak_row):
	i_theta = peak[i][0]
	i_p = peak[i][1]
	i_theta_radians = math.radians( i_theta )
	if (i_theta == 0 or i_theta == 180):
		i_x = i_p / math.cos( i_theta_radians )
		for j in range(0, col):
			if xy_in_range(i_x, j):
				edge_map[i_x][j] = 0
	else:
		for i_x in range(0, row):
			i_y = int( ( i_p - i_x * math.cos( i_theta_radians ) )/ math.sin( i_theta_radians ) )
			if xy_in_range(i_x, i_y):
				edge_map[i_x][i_y] = 0


#plt.imshow(edge_map, cmap='gray')
#plt.show()

#############################################################################################
# Extract line segments
parallel_peak_dict = {}
for key in peak_dict:
	if len( peak_dict[ key ]) < 2: # less than 2 lines, there is no parallelograms
		continue
	else:
		parallel_peak_dict[ key ] = peak_dict[ key ]

# Compute possible parallelogram options
para_gram_options = []
para_keys = list( it.combinations( parallel_peak_dict.keys(), 2) )
for keys in para_keys:
	theta1, theta2 = keys
	p1_list = list( it.combinations( parallel_peak_dict[ theta1 ], 2 ) )
	p2_list = list( it.combinations( parallel_peak_dict[ theta2 ], 2 ) )
	for p1 in p1_list:
		for p2 in p2_list:
			para_gram_options.append( keys + p1 + p2 )

#print( para_gram_options )
# Compute valid parallelogram
for line in para_gram_options:
	theta1 = line[0]
	theta2 = line[1]
	p1_1 = line[2]
	p1_2 = line[3]
	p2_1 = line[4]
	p2_2 = line[5]
	valid_parallelogram



#Saving filtered image to new file
