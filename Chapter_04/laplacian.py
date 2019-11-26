import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
#pip install scikit-image
from skimage import io, exposure, img_as_uint, img_as_float
from skimage.color import rgb2gray


def Lablacian(img):
    kernel_laplace = np.array([np.array([0, -1, 0]), np.array([-1, 4, -1]), np.array([0, -1, 0])])
    G = ndimage.convolve(img, kernel_laplace, mode='reflect')

    return G



img = io.imread('2.jpg')    # Load the image
grayscale = rgb2gray(img)
G=Lablacian(grayscale)
# Adjust the contrast of the filtered image by applying Histogram Equalization
# Adjust the contrast of the filtered image by applying Histogram Equalization
edges_equalized = exposure.equalize_adapthist(G/np.max(np.abs(G)), clip_limit=0.03)

plt.imshow(edges_equalized, cmap=plt.cm.gray)    # plot the edges_clipped
plt.axis('off')
plt.show()