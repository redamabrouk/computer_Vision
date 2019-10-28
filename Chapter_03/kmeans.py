import math 
from PIL import Image
from pylab import *
import matplotlib.cm as cm
import scipy as sp
import random


im = Image.open('test2.jpg').convert('L')
arr = np.asarray(im)

rows,columns = np.shape(arr)

plt.figure()
plt.imshow(arr)
plt.gray()
#User selects the intial seed point
print '\nPlease select the initial seed point'

pseed = plt.ginput(3)
print(pseed)

rand_points=[arr[int(pseed[0][1]),int(pseed[0][0])],arr[int(pseed[1][1]),int(pseed[1][0])],arr[int(pseed[2][1]),int(pseed[2][0])]]
#Intial random centroid po
print(rand_points)

#rand_points = [ random.randint(0, 255) for i in range(3) ]
'''finding the histogram of the image to obtain total number of pixels in each level'''

hist,bins = np.histogram(arr,256,[0,256])

class1_avrage_Center = 0
class2_avrage_Center = 0
class3_avrage_Center = 0
def kmeans(histogram):
	for update_Iteration in range(0,100):
		''' First iteration assign random centroid points '''
		if update_Iteration == 0:
			class1_center_value = rand_points[0]
			class2_center_value = rand_points[1]
			class3_center_value = rand_points[2]

		else:
			#print '\n selecting centroid values'
			class1_center_value = class1_avrage_Center
			class2_center_value = class2_avrage_Center
			class3_center_value = class3_avrage_Center

		#print histogram
		class1_PixelValues = []
		class2_pixelValues = []
		class3_pixelValues = []

		class1PixelsCountArr = []
		class2PixelsCountArr = []
		class3PixelsCountArr = []
		sum1 = 0
		sum2 = 0
		sum3 = 0
		for pixelValue,numberOfPixels in enumerate(histogram):
			#computing absolute distance from each of the cluster and assigning it to a particular cluster based on distance
			# Avarage is calculated using probability formula

			if  (abs(pixelValue - class1_center_value) <  abs(pixelValue - class2_center_value)) and (abs(pixelValue - class1_center_value) <  abs(pixelValue - class3_center_value)):
				class1_PixelValues.append(pixelValue)
				class1PixelsCountArr.append(numberOfPixels)
				sum1 = sum1 + (pixelValue * numberOfPixels)
			elif  abs(pixelValue - class2_center_value) <  abs(pixelValue - class3_center_value):
				class2_pixelValues.append(pixelValue)
				class2PixelsCountArr.append(numberOfPixels)
				sum2 = sum2 + (pixelValue * numberOfPixels)
			else:
				class3_pixelValues.append(pixelValue)
				class3PixelsCountArr.append(numberOfPixels)
				sum3 = sum3 + (pixelValue * numberOfPixels)
	
		class1_avrage_Center = int(sum1)/sum(class1PixelsCountArr)	
		class2_avrage_Center = int(sum2)/sum(class2PixelsCountArr)
		class3_avrage_Center = int(sum3)/sum(class3PixelsCountArr)			
			
	return [class1_PixelValues,class2_pixelValues,class3_pixelValues] 

class1_pixelValues,class2_pixelValues,class3_pixelValues = kmeans(hist)

end = np.zeros((rows,columns))

for i in range(rows):
	for j in range(columns):
		
		if (arr[i][j] in class1_pixelValues):
			end[i][j] = int(0)

		elif (arr[i][j] in class2_pixelValues):
			end[i][j] = int(128)
		else:
			end[i][j] = int(255)

plt.imshow(end, cmap="gray",vmin=0, vmax=255)
plt.colorbar()
plt.show()
