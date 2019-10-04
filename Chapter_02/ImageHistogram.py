import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

h = [1]
x=int(0)

def Hist(img):
   row, col = img.shape 
   h = np.zeros(256)
   for i in range(0,row):
      for j in range(0,col):
         h[img[i,j]] += 1
   
   return h

image = Image.open('historignal.jpg').convert("L")
img = np.asarray(image)
plt.imshow(image,cmap='gray')
plt.show()
h = Hist(img)
x = np.arange(0,256)
plt.bar(x, h, color='b', width=5, align='center')
plt.show()
