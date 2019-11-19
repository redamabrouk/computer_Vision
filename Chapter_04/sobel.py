import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
#pip install scikit-image
from skimage import io, exposure, img_as_uint, img_as_float
from skimage.color import rgb2gray


def sobel_filters(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    Ix = ndimage.filters.convolve(img, Kx)
    Iy = ndimage.filters.convolve(img, Ky)
    
    G = np.hypot(Ix, Iy)
    
    print G.dtype

    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)
   
    return (G, theta)



img = io.imread('2.jpg')    # Load the image
grayscale = rgb2gray(img)
G,T=sobel_filters(grayscale)
# Adjust the contrast of the filtered image by applying Histogram Equalization

plt.imshow(G, cmap=plt.cm.gray)    # plot the edges_clipped
plt.axis('off')
plt.show()