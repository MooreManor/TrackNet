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

def exchange(row, pos1, pos2):
    temp = row[pos1]
    row[pos1] = row[pos2]
    row[pos2] = temp

def mv_first_col(row):
    new_row = [row[-1]] + row[:-1]
    return new_row

if __name__ == '__main__':

    csv_file_list = ['/datasetb/tennis/haluo/csv/26048.csv', '/datasetb/tennis/haluo/csv/28158.csv']
    # csv_file_list = ['/datasetb/tennis/haluo/csv/26048.csv']
    vid_dir = '/datasetb/tennis/haluo/vids/'
    for csv_file_path in csv_file_list:
        vid_file = vid_dir+osp.basename(csv_file_path)[:-4]+'.mp4'
        # 打开视频文件
        video = cv2.VideoCapture(vid_file)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        end_frame = ((frame_count-1)//10)*10

        # 读取 xlsx 文件
        data = pd.read_excel(csv_file_path.replace('.csv', '.xlsx'), sheet_name='Sheet1')
        # 将数据保存为 csv 文件
        data.to_csv(csv_file_path, index=False)
        # with open(csv_file_path, newline='', encoding='gbk') as csvfile:
        cur_frame = 0
        space_data = ['', '', '', '', '', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']
        with open(csv_file_path, newline='') as csvfile, open(osp.dirname(csv_file_path)+f'/{osp.basename(csv_file_path)[:-4]}_output.csv', 'w', newline='') as outfile:
            reader = csv.reader(csvfile)
            first_row = next(reader)  # 跳过第一行
            writer = csv.writer(outfile)
            first_row = mv_first_col(first_row)
            writer.writerow(first_row)
            data = []
            # for row in reader:
            #     writer.writerow(row)
            while True:
                try:
                    # 使用 next() 函数获取下一行数据
                    row = next(reader)
                    while cur_frame<int(row[-1]):
                        space_data[0] = f'{cur_frame}'
                        cur_frame += 10
                        writer.writerow(space_data)
                    if cur_frame == int(row[-1]):
                        cur_frame += 10
                    else:
                        cur_frame = (int(row[-1])//10+1)*10
                    row = mv_first_col(row)
                    writer.writerow(row)

                except StopIteration:
                    # 如果没有下一行数据，则退出循环
                    break

            while cur_frame<=end_frame:
                space_data[0] = f'{cur_frame}'
                cur_frame += 10
                writer.writerow(space_data)