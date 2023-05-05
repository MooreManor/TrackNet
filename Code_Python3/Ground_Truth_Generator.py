import glob
import csv
import numpy
import matplotlib.pyplot as plt
from PIL import Image
import os
from os.path import expanduser

size = 20
#create gussian heatmap 
def gaussian_kernel(variance):
    x, y = numpy.mgrid[-size:size+1, -size:size+1]
    g = numpy.exp(-(x**2+y**2)/float(2*variance))
    return g 


#make the Gaussian by calling the function
variance = 10
gaussian_kernel_array = gaussian_kernel(variance)
#rescale the value to 0-255
gaussian_kernel_array =  gaussian_kernel_array * 255/gaussian_kernel_array[int(len(gaussian_kernel_array)/2)][int(len(gaussian_kernel_array)/2)]
#change type as integer
gaussian_kernel_array = gaussian_kernel_array.astype(int)

#show heatmap 
plt.imshow(gaussian_kernel_array, cmap=plt.get_cmap('gray'), interpolation='nearest')
plt.colorbar()
plt.show()



#create the heatmap as ground truth
images_path = expanduser("~")+'/dataset/tennis/'
dirs = glob.glob(images_path+'data/Clip*')
for index in dirs:
        #################change the path####################################################
        pics = glob.glob(index + "/*.jpg")
        output_pics_path = images_path+'groundtruth/' + os.path.split(index)[-1]
        label_path = index + "/Label.csv"
        ####################################################################################
        
        #check if the path need to be create
        if not os.path.exists(output_pics_path ):
            os.makedirs(output_pics_path)

            
        #read csv file
        with open(label_path, 'r') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            #skip the headers
            next(spamreader, None)  
            
            for row in spamreader:
                    visibility = int(float(row[1]))
                    FileName = row[0]
                    #if visibility == 0, the heatmap is a black image
                    if visibility == 0:
                        heatmap = Image.new("RGB", (1280, 720))
                        pix = heatmap.load()
                        for i in range(1280):
                            for j in range(720):
                                    pix[i,j] = (0,0,0)
                    else:
                        x = int(float(row[2]))
                        y = int(float(row[3]))
                        
                        #create a black image
                        heatmap = Image.new("RGB", (1280, 720))
                        pix = heatmap.load()
                        for i in range(1280):
                            for j in range(720):
                                    pix[i,j] = (0,0,0)
                                    
                        #copy the heatmap on it
                        for i in range(-size,size+1):
                            for j in range(-size,size+1):
                                    if x+i<1280 and x+i>=0 and y+j<720 and y+j>=0 :
                                        temp = gaussian_kernel_array[i+size][j+size]
                                        if temp > 0:
                                            pix[x+i,y+j] = (temp,temp,temp)
                    #save image
                    heatmap.save(output_pics_path + "/" + FileName.split('.')[-2] + ".png", "PNG")
