import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g


def gaussian_filter(image,size=5,sigma=1):
    kernel = gaussian_kernel(size,sigma)
    print kernel
    # fetch the dimensions for iteration over the pixels and weights
    i_width, i_height = image.shape[0], image.shape[1]
    k_width, k_height = kernel.shape[0], kernel.shape[1]

    # prepare the output array
    filtered = np.zeros_like(image)

    # Iterate over each (x, y) pixel in the image ...
    for y in range(i_height):
        for x in range(i_width):
            weighted_pixel_sum = 0
        
            for ky in range(-(k_height / 2), k_height - (size/2+1)):
                for kx in range(-(k_width / 2), k_width - (size/2+1)):
                    pixel = 0
                    pixel_y = y - ky
                    pixel_x = x - kx

                    # boundary check: all values outside the image are treated as zero.
                    # This is a definition and implementation dependent, it's not a property of the convolution itself.
                    if (pixel_y >= 0) and (pixel_y < i_height) and (pixel_x >= 0) and (pixel_x < i_width):
                        pixel = image[pixel_y, pixel_x]

                    # get the weight at the current kernel position
                    # (also un-shift the kernel coordinates into the valid range for the array.)
                    weight = kernel[ky + (k_height / 2), kx + (k_width / 2)]

                    # weigh the pixel value and sum
                    weighted_pixel_sum += pixel * weight

            # finally, the pixel at location (x,y) is the sum of the weighed neighborhood
            filtered[y, x] = weighted_pixel_sum 

    return filtered

img = Image.open("lena.png").convert("L")
plt.imshow(img, cmap=plt.cm.gray)    # plot the edges_clipped
plt.axis('off')
plt.show()
arr = np.array(img)
removed_noise = gaussian_filter(arr,5) 
img = Image.fromarray(removed_noise)
plt.imshow(img, cmap=plt.cm.gray)    # plot the edges_clipped
plt.axis('off')
plt.show()
