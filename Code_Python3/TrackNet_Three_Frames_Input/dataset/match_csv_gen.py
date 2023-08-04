import numpy as np
import cv2
import glob
import itertools
import random
import csv
import os

training_file_name = "match_train_full.csv"
base_path = '/datasetb/tennis'
# csv_path = '/datasetb/tennis/haluo/csv'
# dirs = os.listdir(images_path)
# dirs = [images_path+f'/{x}' for x in dirs]
# dirs = [images_path+f'/26048_sample', images_path+f'/28158_sample']
game_set = ['game1', 'game2', 'game3', 'game4', 'game5', 'game6', 'game7', 'game8', 'game9', 'game10']
game_dir = [base_path+'/'+x for x in game_set]
csv_dir = [glob.glob(x + '/**/*.csv', recursive=True) for x in game_dir]
csv_dir = [item for sublist in csv_dir for item in sublist]
count = 0
with open(training_file_name, 'w') as file:
    file.write("img, img1, img2, ann, hit, bounce, first, last\n")
    for csv_file in csv_dir:
        game_cur = csv_file.split('/')[-3]
        clip_cur = csv_file.split('/')[-2]
        flag = 0
        csv_data = []
        with open(csv_file, 'r') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            # skip the headers
            next(spamreader, None)
            row_data = []
            for row in spamreader:
                count += 1
                visibility = int(float(row[1]))
                FileName = row[0]
                hit = 0
                bounce = 0
                first = 0
                last = 0
                if visibility == 0:
                    pass
                elif int(float(row[4]))==1:
                    hit=1
                elif int(float(row[4]))==2:
                    bounce=1
                row_data = [hit, bounce, first, last, FileName]
                csv_data.append(row_data)
            hit_data = [row[0] for row in csv_data]
            try:
                first_id = hit_data.index(1)
                hit_data.reverse()
                last_id = len(hit_data) - hit_data.index(1) - 1
                csv_data[first_id][2] = 1
                csv_data[last_id][3] = 1
            except:
                pass
            label_path = os.path.dirname(csv_file)
            name_data = [label_path +'/'+row[4] for row in csv_data]

            last_slash_index = label_path.rfind("/")
            gt_path = label_path[:last_slash_index] + "_GroundTruth" + label_path[last_slash_index:]
            gt_data = [gt_path+'/'+row[4][:-4]+'.png' for row in csv_data]

            for i in range(2, len(csv_data)):
                file.write(name_data[i] + "," + name_data[i - 1] + "," + name_data[i - 2] + "," + gt_data[i] + "," + str(csv_data[i][0]) + "," + str(csv_data[i][1]) + ","
                           + str(csv_data[i][2]) + "," + str(csv_data[i][3]) + "\n")








# with open(training_file_name, 'w') as file:
#     file.write("img, img1, img2, ann\n")
#     # for g in range(1, 9):
#     for dir in dirs:
#         annos_path = gt_path + f'/{os.path.basename(dir)}'
#         if not os.path.exists(annos_path):
#             continue
#         images = glob.glob(dir + '/**/*.jpg', recursive=True) + glob.glob(dir + '/**/*.png', recursive=True) + glob.glob(
#             dir + '/**/*.jpeg', recursive=True)
#         images.sort()
#         annotations = glob.glob(annos_path + '/**/*.jpg', recursive=True) + glob.glob(annos_path + '/**/*.png', recursive=True) + glob.glob(
#             annos_path + '/**/*.jpeg', recursive=True)
#         annotations.sort()
#         # images = images[:len(annotations)]
#         # assert len(images) == len(annotations)
#         # for im, seg in zip(images, annotations):
#         #     assert (im.split('/')[-1].split(".")[0] == seg.split('/')[-1].split(".")[0])
#         # count += 2
#         count += 1
#         file_images = [os.path.basename(x) for x in images]
#         # for i in range(2, len(images)):
#         for i in range(1, len(annotations)):
#             # remove image path, get image name
#             # ex: D/Dateset/Clip1/0056.jpg => 0056.jpg
#             file_name = annotations[i].split('/')[-1]
#
#             pid = file_images.index(file_name)
#             # check if file image name same as annotation name
#             assert (images[pid].split('/')[-1].split(".")[0] == annotations[i].split('/')[-1].split(".")[0])
#
#             # write all of images path
#             file.write(images[pid] + "," + images[pid - 1] + "," + images[pid - 2] + "," + annotations[i] + "\n")
#             count += 1


print("Total Count:", count)

file.close()