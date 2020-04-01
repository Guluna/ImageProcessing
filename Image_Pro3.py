
import matplotlib.pyplot as plt
import numpy as np
from skimage import data

def show_image(image, title):
    plt.imshow(image)
    plt.title(title)
    plt.show()


# ###****************************************************##
###  ADDING NOISE TO AN IMAGE  ######
# lets make some noise
# ###****************************************************##
# noisy image is also called salt & pepper image

door = plt.imread('door.jpg')

# Import the module and function
from skimage.util import random_noise

# Add noise to the image
noisy_image = random_noise(door)

# Show original and resulting image
show_image(door, 'Original')
show_image(noisy_image, 'Noisy image')


# ###****************************************************##
###  DENOISE IMAGE  using ######
# TV (total variation) denoise
# Bilateral denoise
# ###****************************************************##


# Import the module and function
from skimage.restoration import denoise_tv_chambolle
# Apply total variation filter denoising
denoised_image = denoise_tv_chambolle(noisy_image, multichannel=True)
# Show the noisy and denoised images
show_image(noisy_image, 'Noisy')
show_image(denoised_image, 'Denoised image using TV method')



# Import bilateral denoising function
from skimage.restoration import denoise_bilateral
# Apply bilateral filter denoising
denoised_image_bilateral = denoise_bilateral(noisy_image, multichannel=True)
# Show original and resulting images
# show_image(landscape_image, 'Noisy image')
show_image(denoised_image_bilateral, 'Denoised image using Bilateral method')

# ###****************************************************##
###  IMAGE RESTORATION ######
# ###****************************************************##

# Set a mask of pixels over the damaged parts of image using inpaint
# inpainting is the process of reconstructing lost or deteriorated parts of images and videos.


# ###****************************************************##
###  IMAGE SEGMENTATION ######
# Simplest technique is thresholding which we already covered (separating foreground from background)
# superpixel is a group of connected pixels with similar colors
# ###****************************************************##

# Import the slic function from segmentation module
from skimage.segmentation import slic
# Import the label2rgb function from color module
from skimage.color import label2rgb
print(data.coffee().shape)
# Obtain the segmentation with 400 regions
segments = slic(data.coffee(), n_segments= 400)
# Put segments on top of original image to compare
segmented_image = label2rgb(segments, data.coffee(), kind='avg')
# Show the segmented image
show_image(segmented_image, "Segmented image, 400 superpixels")
# We reduced the image from 240,000 pixels to 400 regions! Much more computationally efficient for example, face detection machine learning models.


# ###****************************************************##
###  FINDING CONTOUR/BOUNDARY OF AN IMAGE  ######
# input to contour finding fn is a binary image
# Steps: 1- Transform image to 2d grayscale 2- Binarize (black & white) the img 3- use find_contour() fn
# ###****************************************************##

from skimage import color
from skimage.filters import threshold_otsu
from skimage.measure import find_contours

# Construct some test data
x, y = np.ogrid[-np.pi:np.pi:100j, -np.pi:np.pi:100j]
r = np.sin(np.exp((np.sin(x)**3 + np.cos(y)**2)))

# Find contours at a constant value of 0.8
contours = find_contours(r, 0.8)
[print(c.shape) for c in contours]
# Display the image and plot all contours found
fig, ax = plt.subplots()
ax.imshow(r, interpolation='nearest', cmap=plt.cm.gray)

for n, contour in enumerate(contours):
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
# finding contours of specific shape in an img e.g. counting the dots on a dice
dots_contours = [cnt for cnt in contours if np.shape(cnt)[0] < 73]
print("Dice's dots number: {}. ".format(len(dots_contours)))

# ###****************************************************##
###  EDGE DETECTION IN IMAGE  ######
# we have already used sobel but this time we will use Canny which produces better results
# ###****************************************************##

# Import the canny edge detector
from skimage.feature import canny

domino = plt.imread('domino.jpg')
# Convert image to grayscale
domino_binary = color.rgb2gray(domino)

# Apply canny edge detector
canny_edges = canny(domino_binary)

# Show resulting image
show_image(canny_edges, "Edges with Canny")

# ###********************** Corner detection ******************************##

def show_image_with_corners(image, coords, title= 'Corners detected'):
    plt.imshow(image, interpolation='nearest', cmap='gray')
    plt.title(title)
    plt.plot(coords[:, 1], coords[:, 0], '+r', markersize=15)
    # plt.axis()
    plt.show()


# Import the corner detector related functions and module
from skimage.feature import corner_harris, corner_peaks

# Convert image from RGB-3 to grayscale
d = color.rgb2gray(domino)

# Apply the detector  to measure the possible corners
measure_image = corner_harris(d)

# Find the peaks of the corners using the Harris detector
coords = corner_peaks(measure_image, min_distance=2)

# Show original and resulting image with corners detected
show_image(d, "Original")
show_image_with_corners(d, coords)

# ###********************** Face detection ******************************##

from skimage.feature import Cascade
import matplotlib.patches as patches

def show_detected_face(result, detected, title='Face detected'):
    plt.imshow(result)
    img_desc = plt.gca()
    plt.set_cmap('gray')
    plt.title(title)
    plt.axis('off')

    for patch in detected:
        rect = patches.Rectangle((patch['c'], patch['r']),
                           patch['width'],
                           patch['height'],
                           fill=False, color='r',
                           linewidth=2)
        img_desc.add_patch(rect)

    plt.show()

from skimage.feature import Cascade

astronaut = data.astronaut()
# Load the trained file from data
trained_file = data.lbp_frontal_face_cascade_filename()

# Initialize the detector cascade
detector = Cascade(trained_file)

# Detect faces with min and max size of searching window
detected = detector.detect_multi_scale(img = astronaut,
                                       scale_factor=1.2,
                                       step_ratio=1,
                                       min_size=(10,10),
                                       max_size=(200,200))

# Show the detected faces
show_detected_face(astronaut, detected)


# ###********************** Blurry Face  ******************************##

from skimage.filters import gaussian

def get_Face(d, image):
    ''' Extracts the face rectangle from the image using the coordinates of the detected.'''
    # X and Y starting points of the face rectangle
    x, y = d['r'], d['c']
    # The width and height of the face rectangle
    width, height = d['r'] + d['width'], d['c'] + d['height']
    # Extract the detected face
    face = image[x:width, y:height]
    return face

def mergeBlurryFace(original, gaussian_image):
    # X and Y starting points of the face rectangle
    x, y = d['r'], d['c']
    # The width and height of the face rectangle
    width, height = d['r'] + d['width'], d['c'] + d['height']
    original[ x:width, y:height] = gaussian_image
    return original

# Detect the faces
detected = detector.detect_multi_scale(img=astronaut,
                                       scale_factor=1.5, step_ratio=1,
                                       min_size=(20, 20), max_size=(200, 200))
# For each detected face
for d in detected:
    # Obtain the face rectangle from detected coordinates
    face = get_Face(d, astronaut)

    # Apply gaussian filter to extracted face
    blurred_face = gaussian(face, multichannel=True, sigma=10)

    # Merge this blurry face to our final image and show it
    resulting_image = mergeBlurryFace(astronaut, blurred_face)
show_image(resulting_image, "Blurred faces")