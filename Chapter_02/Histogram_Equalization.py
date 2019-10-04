
import math
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

threshold_values = {}
h = [1]
x=int(0)
enhanced_img={}

def Hist(img):
   row, col = img.shape 
   h = np.zeros(256)
   for i in range(0,row):
      for j in range(0,col):
         h[img[i,j]] += 1
   
   return h

def Hist_Eq(img,hmin,hmax):
    row, col = img.shape 
    y = np.zeros((row, col))
    m=255.0/(hmax-hmin)
    for i in range(0,row):
        for j in range(0,col):
            if img[i,j] >= hmin and img[i,j] <= hmax:
                y[i,j] =math.floor(m*(img[i,j]-hmin))
            else:
                y[i,j] = 255
    y=y.astype(int)
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

def periorPropability(start, end):
    w = 0
    for i in range(start, end):
        w += h[i]
    return w

def mean(start, end):
    m = 0
    w = periorPropability(start, end)
    for i in range(start, end):
        m += h[i] * i
    
    return m/float(w)

def variance(start, end):
    v = 0
    m = mean(start, end)
    w = periorPropability(start, end)
    for i in range(start, end):
        v += ((i - m) **2) * h[i]
    v /= w
    return v

def threshold(h):
    cnt = countPixel(h)
    for i in range(1, len(h)):
       	variance_1 = variance(0, i)
        w1 = periorPropability(0, i) / float(cnt)
        mu1 = mean(0, i)
        variance_2 = variance(i, len(h))
        w2 = periorPropability(i, len(h)) / float(cnt)
        mu2 = mean(i, len(h))
        variance_within_class = w1 * (variance_1) + w2 * (variance_2)
        if not math.isnan(variance_within_class):
            threshold_values[i] = variance_within_class

def get_optimal_threshold():
    min_variance_within_class = min(threshold_values.itervalues())
    print min_variance_within_class
    optimal_threshold = [k for k, v in threshold_values.iteritems() 
    if v == min_variance_within_class]
    print optimal_threshold
    print 'optimal threshold', optimal_threshold[0]
    return optimal_threshold[0]


image = Image.open('historignal.jpg').convert("L")
img = np.asarray(image)
plt.imshow(image,cmap='gray')
plt.show()
h = Hist(img)
x = np.arange(0,256)

plt.bar(x, h, color='b', width=5, align='center')
plt.show()
newImage=Hist_Eq(img,102,148)
img = np.asarray(newImage)
plt.imshow(newImage,cmap='gray')
plt.show()
h = Hist(img)
x = np.arange(0,256)
plt.bar(x, h, color='b', width=5, align='center')
plt.show()
