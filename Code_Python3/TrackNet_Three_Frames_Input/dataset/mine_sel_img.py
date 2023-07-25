import csv
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import os.path as osp
import numpy as np
import glob
import cv2
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
# from utils import add_text_to_frame
import pandas as pd
import shutil


def read_csv(csv_file_path):
    # 定义 CSV 文件路径和字段名
    with open(csv_file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # 跳过第一行
        data = []
        for row in reader:
            if row[1]=='':
                frame = int(row[0])
                data.append([frame, None, None, None, None, None, None])
            else:
                frame = int(row[0])
                x, y = int(float(row[1])), int(float(row[2]))
                hit = int(row[5])
                bounce = int(row[7])
                start = int(row[11])
                end = int(row[13])
                data.append([frame, x, y, hit, bounce, start, end])
    # 将数据转换为numpy数组
    data = np.array(data)
    return data


if __name__ == '__main__':
    csv_file_list = ['/datasetb/tennis/haluo/csv/26048_output.csv']
    vid_path = '/datasetb/tennis/haluo/imgs/'
    for csv_file_path in csv_file_list:

        vid_name = osp.basename(csv_file_path)[:-4]
        data = read_csv(csv_file_path)
        for i, pos in enumerate(data):
            img_path = vid_path + vid_name[:-7]
            dst_path = img_path + '_tmp'
            if i == 0:
                src_file = img_path + "/{:03d}/".format(int(pos[0]) // 1000) + "{:06d}".format(int(pos[0])) + '.png'
                dst_file = dst_path + "/{:03d}/".format(int(pos[0]) // 1000) + "{:06d}".format(int(pos[0])) + '.png'
                os.makedirs(osp.dirname(dst_file), exist_ok=True)
                shutil.copy2(src_file, dst_file)
            else:
                for j in range(3):
                    src_file = img_path + "/{:03d}/".format((int(pos[0])-j) // 1000) + "{:06d}".format((int(pos[0])-j)) + '.png'
                    dst_file = dst_path + "/{:03d}/".format((int(pos[0])-j) // 1000) + "{:06d}".format((int(pos[0])-j)) + '.png'
                    os.makedirs(osp.dirname(dst_file), exist_ok=True)
                    shutil.copy2(src_file, dst_file)
