# Parallelograms-Detection
## Project description: 
Parallelograms appear frequently in images that contain man-made objects. They often correspond to the projections of rectangular surfaces when viewed at an angle that is not perpendicular to the surfaces. In this project, you are to design and implement a program that can detect parallelograms of all sizes in an image.

Your program will consist of three steps: 
1. detect edges using the Sobel’s operator, 
2. detect straight line segments using the Hough Transform, and 
3. detect parallelograms from the straight-line segments detected in step (2). In step (1), compute edge magnitude using the formula below and then normalize the magnitude values to lie within the range [0,255]. Next, manually choose a threshold value to produce a binary edge map.

Edge Magnitude = \sqrt{Gx^2 + Gy^2}(square root of G x squared plus G y squared end root, Gx and Gy are the horizontal and vertical gradients, respectively.)

## Input
The test images that will be provided to you are in color so you will need to convert them into grayscale images by using the formula luminance = 0.30R + 0.59G + 0.11B, where R, G, and B, are the red, green, and blue components. Test images in both JPEG band RAW image formats will be provided. In the RAW image format, the red, green, and blue components of the pixels are recorded in an interleaved manner, occupying one byte per color component per pixel (See description below).  The RAW image format does not contain any header bytes.

Python, C/C++, Matlab and Java are the recommended programming languages. If you intend to use a different language, send me an email first. You are not allowed to use any built-in library functions for any of the steps that you are required to implement.

<!--- --->
:information_desk_person:
:information_desk_person:
:information_desk_person:
:information_desk_person:
:information_desk_person:

* You are not allowed to use the Convolution or Cross-Correlation built-in library function of any programming language. You have to write your own function. 
* You are allowed to use  Read, Write, and Display images library functions. 
* There are 3 sets of test images attached. Each image is available in both .jpg and .raw formats. See Project Description for a description of the .raw format. TestImage1c and Testimage2c are of dimension 756 rows x 1008 columns, and Testimage3 has dimension 413 rows x 550 columns. 
