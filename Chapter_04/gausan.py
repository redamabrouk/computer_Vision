import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage import io, exposure, img_as_uint, img_as_float
from skimage.color import rgb2gray


def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g



img = io.imread('lena.png')    # Load the image
grayscale = rgb2gray(img)
gk=gaussian_kernel(5,1.4)
plt.imshow(grayscale, cmap=plt.cm.gray)    # plot the edges_clipped
plt.axis('off')
plt.show()
smoothimg = ndimage.filters.convolve(grayscale, gk)
# Adjust the contrast of the filtered image by applying Histogram Equalization

plt.imshow(smoothimg, cmap=plt.cm.gray)    # plot the edges_clipped
plt.axis('off')
plt.show()