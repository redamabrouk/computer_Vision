import numpy
from PIL import Image
import matplotlib.pyplot as plt


def median_filter(data, filter_size):
    
    temp = []
    # for size=5
    # indexer = 2
    indexer = filter_size // 2
    data_final = []
    data_final = numpy.zeros((len(data),len(data[0])))
    #for each ith row
    for i in range(len(data)):
        #for Each jth column    
        for j in range(len(data[0])):
            # z= 0, 1, 2, 3, 4, 5
            for z in range(filter_size):
                if i + z - indexer < 0 or i + z - indexer > len(data) - 1:
                    for c in range(filter_size):
                        temp.append(0)
                else:
                    if j + z - indexer < 0 or j + indexer > len(data[0]) - 1:
                        temp.append(0)
                    else:
                        for k in range(filter_size):
                            temp.append(data[i + z - indexer][j + k - indexer])

            temp.sort()
            #data_final[i][j] =temp[-1]

            data_final[i][j] = temp[len(temp) // 2]
            temp = []
    return data_final

img = Image.open("noisyimg.png").convert("L")
plt.imshow(img, cmap=plt.cm.gray)    # plot the edges_clipped
plt.axis('off')
plt.show()
arr = numpy.array(img)
removed_noise = median_filter(arr, 5) 
img = Image.fromarray(removed_noise)
plt.imshow(img, cmap=plt.cm.gray)    # plot the edges_clipped
plt.axis('off')
plt.show()
