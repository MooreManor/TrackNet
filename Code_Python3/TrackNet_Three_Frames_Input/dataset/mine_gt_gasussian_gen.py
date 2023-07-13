# -*- coding: utf-8 -*-
import glob
import csv
import numpy
import matplotlib.pyplot as plt
from PIL import Image
import os
from os.path import expanduser

# 1280*720
# size = 20
# 1920*1080
size = 30


# create gussian heatmap
def gaussian_kernel(variance):
    x, y = numpy.mgrid[-size:size + 1, -size:size + 1]
    g = numpy.exp(-(x ** 2 + y ** 2) / float(2 * variance))
    return g


# make the Gaussian by calling the function
variance = 10
gaussian_kernel_array = gaussian_kernel(variance)
# rescale the value to 0-255
gaussian_kernel_array = gaussian_kernel_array * 255 / gaussian_kernel_array[int(len(gaussian_kernel_array) / 2)][
    int(len(gaussian_kernel_array) / 2)]
# change type as integer
gaussian_kernel_array = gaussian_kernel_array.astype(int)

# show heatmap
# plt.imshow(gaussian_kernel_array, cmap=plt.get_cmap('gray'), interpolation='nearest')
# plt.colorbar()
# plt.show()

# create the heatmap as ground truth
# images_path = expanduser("~")+'/dataset/tennis/'
images_path = '/datasetb/tennis/haluo/imgs'
csv_path = '/datasetb/tennis/haluo/csv'
gt_path = '/datasetb/tennis/haluo/GroundTruth'
# dirs = glob.glob(images_path+'data/Clip*')
dirs = os.listdir(images_path)
dirs = [images_path+f'/{x}' for x in dirs]
# dirs = glob.glob(images_path + '/**/*.png', recursive=True)
for index in dirs:
    #################change the path####################################################
    pics = glob.glob(images_path + '/**/*.png', recursive=True)
    pic_name = [int(os.path.basename(x)[:-4]) for x in pics]
    vid_name = os.path.basename(index)
    output_pics_path = gt_path + '/' + vid_name
    label_path = csv_path + f"/{vid_name}.csv"
    if not os.path.exists(label_path):
        continue
    ####################################################################################

    # check if the path need to be create
    if not os.path.exists(output_pics_path):
        os.makedirs(output_pics_path)

    # read csv file
    with open(label_path, 'r', encoding='gbk') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        # skip the headers
        next(spamreader, None)

        # count = 0
        for row in spamreader:
            # count += 1
            # if count == 100:
            #     print(row)
            visibility = 0 if len(row)==1 else 1
            FileName = int(row[0])-1
            # if visibility == 0, the heatmap is a black image
            if visibility == 0:
                heatmap = Image.new("RGB", (1920, 1080))
                pix = heatmap.load()
                for i in range(1920):
                    for j in range(1080):
                        pix[i, j] = (0, 0, 0)
            else:
                x = int(float(row[1]))
                y = int(float(row[2]))

                # create a black image
                heatmap = Image.new("RGB", (1920, 1080))
                pix = heatmap.load()
                for i in range(1920):
                    for j in range(1080):
                        pix[i, j] = (0, 0, 0)

                # copy the heatmap on it
                for i in range(-size, size + 1):
                    for j in range(-size, size + 1):
                        if x + i < 1920 and x + i >= 0 and y + j < 1080 and y + j >= 0:
                            temp = gaussian_kernel_array[i + size][j + size]
                            if temp > 0:
                                pix[x + i, y + j] = (temp, temp, temp)
            # save image
            pic_ind = pic_name.index(FileName)
            output_file = output_pics_path + '/' +'/'.join(pics[pic_ind].split('/')[-2:])
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            heatmap.save(output_file, "PNG")
