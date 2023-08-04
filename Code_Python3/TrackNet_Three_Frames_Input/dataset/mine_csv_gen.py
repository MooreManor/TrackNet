# 4.Output All of training data name to cvs file for new labeling
import numpy as np
import cv2
import glob
import itertools
import random
import csv
import os

# training_file_name = "eval.csv"
training_file_name = "mine_train.csv"
images_path = '/datasetb/tennis/haluo/imgs'
# csv_path = '/datasetb/tennis/haluo/csv'
gt_path = '/datasetb/tennis/haluo/GroundTruth'
dirs = os.listdir(images_path)
# dirs = [images_path+f'/{x}' for x in dirs]
dirs = [images_path+f'/26048_sample', images_path+f'/28158_sample']
count = 0
with open(training_file_name, 'w') as file:
    file.write("img, img1, img2, ann, hit, bounce, first, last\n")
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

    # for g in range(1, 11):
    #     #################change the path####################################################
    #     LabelingData_dir = '/datasetb/tennis/game' + str(g)
    #     ####################################################################################
    #     list = os.listdir(LabelingData_dir)  # dir is your directory path
    #     number_clips = len(list)
    #     for index in range(1, number_clips + 1):
    #         #################change the path####################################################
    #         images_path = "/datasetb/tennis/game" + str(g) + "/Clip" + str(index) + "/"
    #         annos_path = "/datasetb/tennis/game" + str(g) + "_GroundTruth/Clip" + str(
    #             index) + "/"
    #         ####################################################################################
    #
    #         images = glob.glob(images_path + "*.jpg") + glob.glob(images_path + "*.png") + glob.glob(
    #             images_path + "*.jpeg")
    #         images.sort()
    #         annotations = glob.glob(annos_path + "*.jpg") + glob.glob(annos_path + "*.png") + glob.glob(
    #             annos_path + "*.jpeg")
    #         annotations.sort()
    #
    #         # check if annotation counts equals to image counts
    #         assert len(images) == len(annotations)
    #         for im, seg in zip(images, annotations):
    #             assert (im.split('/')[-1].split(".")[0] == seg.split('/')[-1].split(".")[0])
    #
    #         # with open(images_path + "Label.csv", 'r') as csvfile:
    #         #     spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    #         #     # skip the headers
    #         #     next(spamreader, None)
    #
    #         count += 2
    #
    #         # output all of images path, 0000.jpg & 0001.jpg cant be used as input, so we have to start from 0002.jpg
    #         for i in range(2, len(images)):
    #             # remove image path, get image name
    #             # ex: D/Dateset/Clip1/0056.jpg => 0056.jpg
    #             file_name = images[i].split('/')[-1]
    #
    #             # check if file image name same as annotation name
    #             assert (images[i].split('/')[-1].split(".")[0] == annotations[i].split('/')[-1].split(".")[0])
    #
    #             # write all of images path
    #             file.write(images[i] + "," + images[i - 1] + "," + images[i - 2] + "," + annotations[i] + "\n")
    #             count += 1

print("Total Count:", count)

file.close()