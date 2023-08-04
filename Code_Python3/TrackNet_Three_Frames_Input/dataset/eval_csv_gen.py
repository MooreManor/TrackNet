import numpy as np
import cv2
import glob
import itertools
import random
import csv
import os

training_file_name = "eval.csv"
base_path = '/datasetb/tennis'
csv_path = '/datasetb/tennis/haluo/csv'
img_path = '/datasetb/tennis/haluo/imgs'
# dirs = os.listdir(images_path)
# dirs = [images_path+f'/{x}' for x in dirs]
# dirs = [images_path+f'/26048_sample', images_path+f'/28158_sample']
vid_names = ['16811385934162', '168000453563685', '168113494209423', '168113658899197', '168113862372680', '168114225412797', '168114233854976', '168114343892521', '168118847101172', '1680102042897106']
csv_dir = [csv_path + '/' + file+'.csv' for file in vid_names]
count = 0
with open(training_file_name, 'w') as file:
    file.write("img, img1, img2, ann, hit, bounce, first, last\n")
    for csv_file in csv_dir:
        csv_data = []
        with open(csv_file, 'r', encoding='gbk') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            # skip the headers
            next(spamreader, None)
            row_data = []
            for row in spamreader:
                count += 1
                visibility=1
                if len(row)==1:
                    visibility = 0
                FileName = int(float(row[0]))-1
                FileName = img_path + '/' + os.path.basename(csv_file)[:-4] + "/000/{:06d}.png".format(FileName)
                hit = 0
                bounce = 0
                first = 0
                last = 0
                if visibility == 0:
                    row_data = [hit, bounce, first, last, FileName]
                    csv_data.append(row_data)
                    continue
                if int(float(row[3]))==1:
                    hit=1
                if int(float(row[4]))==1:
                    bounce=1
                if int(float(row[6]))==1:
                    first=1
                if int(float(row[7]))==1:
                    last=1
                row_data = [hit, bounce, first, last, FileName]
                csv_data.append(row_data)

            label_path = os.path.dirname(csv_file)
            name_data = [row[4] for row in csv_data]
            gt_data = [name.replace('imgs', 'GroundTruth') for name in name_data]

            for i in range(2, len(csv_data)):
                file.write(name_data[i] + "," + name_data[i - 1] + "," + name_data[i - 2] + "," + gt_data[i] + "," + str(csv_data[i][0]) + "," + str(csv_data[i][1]) + ","
                           + str(csv_data[i][2]) + "," + str(csv_data[i][3]) + "\n")


print("Total Count:", count)

file.close()