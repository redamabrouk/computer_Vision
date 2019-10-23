# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 21:38:08 2019

@author: DrReda
"""

import math
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

def Hist(img):
   row, col = img.shape 
   y = np.zeros(256)
   for i in range(0,row):
      for j in range(0,col):
         y[img[i,j]] += 1
   x = np.arange(0,256)
   plt.bar(x, y, color='b', width=5, align='center', alpha=0.25)
   plt.show()
   return y


def regenerate_img(img, threshold):
    row, col = img.shape 
    y = np.zeros((row, col))
    for i in range(0,row):
        for j in range(0,col):
            if img[i,j] >= threshold:
                y[i,j] = 255
            else:
                y[i,j] = 0
    return y

image = Image.open('historignal.jpg').convert("L")
img = np.asarray(image)

h = Hist(img)

res = regenerate_img(img, 120)
plt.imshow(res,cmap='gray')
plt.show()
plt.imsave("otsu.jpg",res,cmap='gray')
