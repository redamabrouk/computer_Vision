"""
Created on Mon Oct 30 12:41:30 2017
@author: mohabmes
"""

import math
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


threshold_values = {}
h = [1]


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


   
def countPixel(h):
    cnt = 0
    for i in range(0, len(h)):
        if h[i]>0:
           cnt += h[i]
    return cnt


def wieght(s, e):
    w = 0
    for i in range(s, e):
        w += h[i]
    return w


def mean(s, e):
    m = 0
    w = wieght(s, e)
    for i in range(s, e):
        m += h[i] * i
    
    return m/float(w)


def variance(start, end):
    v = 0
    m = mean(start, end)
    w = wieght(start, end)
    for i in range(start, end):
        v += ((i - m) **2) * h[i]
    v /= w
    return v
            

def threshold(h):
    cnt = countPixel(h)
    for i in range(1, len(h)):
        class1_variance = variance(0, i)
        class1_wieght = wieght(0, i) / float(cnt)
        class1_mean = mean(0, i)
        
        class2_variance = variance(i, len(h))
        class2_wieght = wieght(i, len(h)) / float(cnt)
        class2_mean = mean(i, len(h))
        
        Within_class_variance = class1_wieght * (class1_variance) + class2_wieght * (class2_variance)
        between_class_variance = class1_wieght * class2_wieght * (class1_mean - class2_mean)**2
        
        fw = open("trace.txt", "a")
        fw.write('T='+ str(i) + "\n")

        fw.write('class1_wieght='+ str(class1_wieght) + "\n")
        fw.write('class1_mean='+ str(class1_mean) + "\n")
        fw.write('class1_variance='+ str(class1_variance) + "\n")
        
        fw.write('class2_wieght='+ str(class2_wieght) + "\n")
        fw.write('class2_mean='+ str(class2_mean) + "\n")
        fw.write('class2_variance='+ str(class2_variance) + "\n")

        fw.write('within class variance='+ str(Within_class_variance) + "\n")
        fw.write('between class variance=' + str(between_class_variance) + "\n")
        fw.write("\n")
        
        if not math.isnan(Within_class_variance):
            threshold_values[i] = Within_class_variance


def get_optimal_threshold():
    min_Within_class_variance = min(threshold_values.itervalues())
    optimal_threshold = [k for k, v in threshold_values.iteritems() if v == min_Within_class_variance]
    print 'optimal threshold', optimal_threshold[0]
    return optimal_threshold[0]


image = Image.open('historignal.jpg').convert("L")
img = np.asarray(image)

h = Hist(img)
threshold(h)
op_thres = get_optimal_threshold()

res = regenerate_img(img, op_thres)
plt.imshow(res,cmap='gray')
plt.show()
plt.imsave("otsu.jpg",res,cmap='gray')
