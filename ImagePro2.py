import matplotlib.pyplot as plt
from skimage import color, data

def show_image(image, title):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.show()



# ###************************ Using Filters for smoothing (Blurring via guassian) & Edge detection (via sobel)  ****************************##
# These filters only work with Grayscale images

# detecting the edges in the image using sobel
# Import the filters module and sobel function
from skimage.filters import sobel
cell = data.cell()
cell_image_gray = color.rgb2gray(cell)

# Apply edge detection filter
edge_sobel = sobel(cell_image_gray)

# Show original and resulting image to compare
plt.imshow(cell)
plt.show()
plt.imshow(edge_sobel, cmap='gray')
plt.show()


# ###************************ Blurring (reducing the sharpness of an image) to reduce noise  ****************************##
# blurring the edges in the image using guassian

# Import Gaussian filter
from skimage.filters import gaussian
# Apply filter
gaussian_image = gaussian(data.brick(), multichannel=True)      # set multichannel to true for RGB pics

# Show original and resulting image to compare
plt.imshow(data.brick())
plt.show()
plt.imshow(gaussian_image, cmap='gray')
plt.show()



# ###************************ Contrast Enhancement  ****************************##
# Contrast is the diff bw max & min intensities in histogram of image. So for higher contrast, the histogram of
# intensities is more spread out. An image of low contrast has a very small range of intensities.

# * Histogram equalization spreads out the most frequent global intensity values in img

# required module for contrast operations
from skimage import exposure

chest_img = plt.imread('chest.jpg')
# Show original x-ray image and its histogram
plt.imshow(chest_img, cmap='gray')
plt.title('Original x-ray')
plt.show()

plt.title('Histogram of image')
plt.hist(chest_img.ravel(), bins=256)
plt.show()

# Use histogram equalization to improve the contrast
xray_image_eq =  exposure.equalize_hist(chest_img)

# Show the resulting image
plt.imshow(xray_image_eq, cmap='gray')
plt.title('Resulting image')
plt.show()


# * Adaptive Eqaulization amplies the most frequent local intensity values i img so it usually gives more natural results
# Local details can therefore be enhanced even in regions that are darker or lighter than the rest of the image.

# Import the necessary modules
from skimage import data, exposure

# Load the image
original_image = data.coffee()

# Apply the adaptive equalization on the original image
adapthist_eq_image = exposure.equalize_adapthist(original_image, clip_limit=0.03)  # higher the clip_limit, higher the contrast

# Compare the original image to the equalized
show_image(original_image, 'Original')
show_image(adapthist_eq_image, 'After applying Adaptive equalization for contrast enhancement')


# skimage's transform module
# * Rotating images clock-wise (angle in - degrees) or anti-clockwise (angle in + degrees) from the center using skimage.transform
# * Rescaling to adjust size of image e.g. to pass 1/4 as 2nd parameter to make image 4x smaller than its original size
# multichannel = true means colored image, aliasing pixelation (square boxes) in image
# * Resize takes a tuple of (height, width) of output image

# Import the module and the rotate and rescale functions
from skimage.transform import rotate, rescale
# Rotate the image 90 degrees clockwise
rotated_cat_image = rotate(data.coffee(), -90)
# Rescale with anti aliasing
rescaled_with_aa = rescale(rotated_cat_image, 1/4, anti_aliasing=True, multichannel=True)
# Rescale without anti aliasing
rescaled_without_aa = rescale(rotated_cat_image, 1/4, anti_aliasing=False, multichannel=True)
# Show the resulting images
show_image(rescaled_with_aa, "Transformed with anti aliasing")
show_image(rescaled_without_aa, "Transformed without anti aliasing")
# rotated and rescaled the image.
# Seems like the anti aliasing filter prevents the poor pixelation effect to happen, making it look better but also less sharp.



# resizing an image to make it larger? This usually results in loss of quality, with the enlarged image looking blurry.
# Import the module and function to enlarge images
from skimage.transform import resize
# Import the data module
from skimage import data
# Load the image from data
rocket_image = data.rocket()
height = rocket_image.shape[0]*3
width = rocket_image.shape[1]*3
# Enlarge the image so it is 3 times bigger
enlarged_rocket_image = resize(rocket_image, (height, width), anti_aliasing=True)
# Show original and resulting image
show_image(rocket_image, 'original')
show_image(enlarged_rocket_image, "3 times enlarged image")
# The image went from being 600 pixels wide to over 2500 and it still does not look poorly pixelated.


# MORPHOLOGICAL FILTERING OPERATIONS (usually works best on binary image but might work on grayscale as well)
# * Dilation: addes pixels to image boundary
# * erosion: removes pixels from img boundary

###### Morphology not working
# # Import the morphology module
# from skimage import morphology
# binary_img = plt.imread('Binary_coins.jpg')
# # Obtain the eroded shape
# eroded_image_shape = morphology.binary_erosion(binary_img)
# # See results
# show_image(binary_img, 'Original')
# show_image(eroded_image_shape, 'Eroded image')
###### Morphology not working


