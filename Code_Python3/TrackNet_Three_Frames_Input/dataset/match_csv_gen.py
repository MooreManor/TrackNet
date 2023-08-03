import numpy as np
import cv2
import glob
import itertools
import random
import csv
import os

training_file_name = "match_train.csv"
base_path = '/datasetb/tennis'
# csv_path = '/datasetb/tennis/haluo/csv'
gt_path = '/datasetb/tennis/haluo/GroundTruth'
# dirs = os.listdir(images_path)
# dirs = [images_path+f'/{x}' for x in dirs]
# dirs = [images_path+f'/26048_sample', images_path+f'/28158_sample']
game_set = ['game1', 'game2', 'game3', 'game4', 'game5', 'game6', 'game7', 'game8', 'game9', 'game10']


count = 0
with open(training_file_name, 'w') as file:
    file.write("img, img1, img2, ann\n")
    # for g in range(1, 9):
    for dir in dirs:
        annos_path = gt_path + f'/{os.path.basename(dir)}'
        if not os.path.exists(annos_path):
            continue
        images = glob.glob(dir + '/**/*.jpg', recursive=True) + glob.glob(dir + '/**/*.png', recursive=True) + glob.glob(
            dir + '/**/*.jpeg', recursive=True)
        images.sort()
        annotations = glob.glob(annos_path + '/**/*.jpg', recursive=True) + glob.glob(annos_path + '/**/*.png', recursive=True) + glob.glob(
            annos_path + '/**/*.jpeg', recursive=True)
        annotations.sort()
        # images = images[:len(annotations)]
        # assert len(images) == len(annotations)
        # for im, seg in zip(images, annotations):
        #     assert (im.split('/')[-1].split(".")[0] == seg.split('/')[-1].split(".")[0])
        # count += 2
        count += 1
        file_images = [os.path.basename(x) for x in images]
        # for i in range(2, len(images)):
        for i in range(1, len(annotations)):
            # remove image path, get image name
            # ex: D/Dateset/Clip1/0056.jpg => 0056.jpg
            file_name = annotations[i].split('/')[-1]

            pid = file_images.index(file_name)
            # check if file image name same as annotation name
            assert (images[pid].split('/')[-1].split(".")[0] == annotations[i].split('/')[-1].split(".")[0])

            # write all of images path
            file.write(images[pid] + "," + images[pid - 1] + "," + images[pid - 2] + "," + annotations[i] + "\n")
            count += 1


print("Total Count:", count)

file.close()