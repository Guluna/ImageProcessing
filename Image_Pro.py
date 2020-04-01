# Digital image is an array of square-shaped pixels. Each pixel can be represented by an intensity (numerical #) in form of matrix
# we will use scikit-image for this tutorial

from skimage import data, color
import numpy as np
import matplotlib.pyplot as plt


coffee = data.coffee()
print(coffee.shape)     # 3D
coins = data.coins()
print(coins.shape)      # 2D


###************************ RGB to Grayscale ****************************##

# Import the modules from skimage
from skimage import data, color
import matplotlib.pyplot as plt


def show_image(image, title='Image', cmap_type='gray'):
    plt.imshow(image, cmap=cmap_type)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Load the rocket image
rocket = data.rocket()
# Convert the image to grayscale
gray_scaled_rocket = color.rgb2gray(rocket)
# Show the original image
show_image(rocket, 'Original RGB image')
# Show the grayscale image
show_image(gray_scaled_rocket, 'Grayscale image')


###************************ Flipping ****************************##
# ndarray = multi-dimensional array
# shape of 3D image returns (Height, Width, ColorChannels). If e.g. shape of image is (426,640,3) then total # of pixels
# in image are 426*640*3=817920
#
door = plt.imread('door.jpg')
# Flip the image vertically
vertical_flip = np.flipud(door)
# Flip the image horizontally
horizontal_flip = np.fliplr(door)

# Show the resulting images
plt.imshow(vertical_flip)
plt.title('Vertically flipped')
plt.show()
plt.imshow(horizontal_flip)
plt.title('Horizontally flipped')
plt.show()


###************************ Computing histograms of RGB channels for image analysis ****************************##

# Both perform the same fn but the Differences between Flatten() and Ravel() are:
# a.ravel() is faster & return reference to original array
# a.flatten() is slower & returns a copy of original array


# Obtain the red channel
red_channel = door[:, :, 0]
# Plot the red histogram with bins in a range of 256
plt.hist(red_channel.ravel(), bins=256)         # spike at just the end of hist suggests there is not much red color involved in this image
# Set title and show
plt.title('Red Histogram')
plt.show()

red_flower = plt.imread('red_flower.jpg')
red_channel_flower = red_flower[:, :, 0]
# Plot the red histogram with bins in a range of 256
plt.hist(red_channel_flower.ravel(), bins=256)         # a lot more red color involved in this image
# Set title and show
plt.title('Red Histogram For flower pic')
plt.show()


###************************ Thresholding ****************************##
# partitioning an image into background & foreground by making it binary
# It is simplest method of image segmentation, works best in high contrast images
# only works on grayscale images so must convert the image to grayscale 1st

# Psuedocode for thresholding
#     If
#     f(x, y) > Threshold value
#     then
#     f(x, y) = 0
# else
#     f(x, y) = 255

# Import the otsu threshold function
# In Otsu Thresholding, a value of the threshold isn’t chosen but is determined automatically

from skimage.filters import threshold_otsu

chess = plt.imread('chess.jpg')
# Make the image grayscale using rgb2gray
chess_pieces_image_gray = color.rgb2gray(chess)
# Obtain the optimal threshold value with otsu
thresh = threshold_otsu(chess_pieces_image_gray)
# Apply thresholding to the image
binary = chess_pieces_image_gray > thresh
# Show the image
plt.imshow(binary, cmap='gray')
plt.title('Thresholded image')
plt.show()




# trying global & local thresholding & see which works better on this image
page = plt.imread('handwriting.jpg')
page_image_gray = color.rgb2gray(page)

# Import the Global threshold function
from skimage.filters import threshold_otsu
# Obtain the optimal otsu global thresh value
global_thresh = threshold_otsu(page_image_gray)
# Obtain the binary image by applying global thresholding
binary_global = page_image_gray > global_thresh
# Show the binary image obtained
plt.imshow(binary_global, cmap='gray')
plt.title('Global Thresholded image')
plt.show()



# ###************************ threshold_local()  ****************************##
# Import the local threshold function
# local threshold should only be used if the image has a wide variation of background intensity

from skimage.filters import threshold_local
# Set the block size to 35
block_size = 35     # blockSize = Size of a pixel neighborhood that is used to calculate a threshold value for the pixel: 3, 5, 7, and so on.
# Obtain the optimal local thresholding
local_thresh = threshold_local(page_image_gray, block_size)
# Obtain the binary image by applying local thresholding
binary_local = page_image_gray > local_thresh
# Show the binary image
plt.imshow(binary_local.astype('uint8'), cmap='gray')
plt.title('Local Thresholded image')
plt.show()      # just producing a completely black picture
#
# local threshold should only be used if the image has a wide variation of background intensity e.g.
vary_bg = plt.imread('varying_bg.jpg')
vary_bg_gray = color.rgb2gray(vary_bg)
block_size = 35     # blockSize = Size of a pixel neighborhood that is used to calculate a threshold value for the pixel: 3, 5, 7, and so on.
local_thresh = threshold_local(vary_bg_gray, block_size)
binary_local = vary_bg_gray > local_thresh
plt.imshow(binary_local.astype('uint8'), cmap='gray')
plt.title('Local Thresholded image')
plt.show()




# ###************************ try_all_threshold()  ****************************##
# comparing the outputs of different global thresholding methods. scikit-image provides us with a function to check
# multiple methods and see for ourselves what the best option is.

# Import the try all function
from skimage.filters import try_all_threshold
# Import the rgb to gray convertor function
from skimage.color import rgb2gray
# Turn the image to grayscale
grayscale = rgb2gray(door)

# Use the try all method on the grayscale image
fig, ax = try_all_threshold(grayscale, verbose=False)       # "Mean" method performed the best whereas "Triangle" performed the worst
# Show the resulting plots
plt.show()

