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
from utils import add_text_to_frame

def read_csv(csv_file_path):
    # 定义 CSV 文件路径和字段名
    with open(csv_file_path, newline='', encoding='gbk') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # 跳过第一行
        data = []
        for row in reader:
            if len(row)==1:
                data.append([None, None, None, None, None, None])
            else:
                x, y = int(float(row[1])), int(float(row[2]))
                hit = int(row[3])
                bounce = int(row[4])
                start = int(row[6])
                end = int(row[7])
                data.append([x, y, hit, bounce, start, end])
    # 将数据转换为numpy数组
    data = np.array(data)
    return data

if __name__ == '__main__':
    csv_file_list = glob.glob('/datasetb/tennis/haluo/csv/*.csv')
    vid_path = '/datasetb/tennis/haluo/imgs/'

    # 设置视频的帧率、宽度和高度
    fps = 30
    width = 1920
    height = 1080

    # 创建视频编写器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 可以根据需要更改视频格式

    csv_file_list = [csv_file_list[6]]
    show_frame = 10
    for csv_file in csv_file_list:
        out = cv2.VideoWriter('../debug.mp4', fourcc, fps, (width, height))
        data = read_csv(csv_file)
        vid_name = osp.basename(csv_file)[:-4]
        img_path = vid_path+vid_name+'/000/'
        for i, pos in enumerate(data):
            img_name = img_path+"{:06d}".format(i)+'.png'
            img = cv2.imread(img_name)

            # if 1 in data[max(0, int(i-show_frame/2)): min(len(img_name), int(i+show_frame/2+1)), 2]:
            if data[i, 0] != None and 1 in data[max(0, int(i-show_frame)): i+1][:, 2]:
                # img = add_text_to_frame(img, f"Hit:({int(data[i][0])},{int(data[i][1])})",
                img = add_text_to_frame(img, f"Hit", position=(800, 150), font_scale=1.5)
            if data[i, 0] != None and data[i,3]==1:
                cv2.circle(img, data[i, :2], 10, (255, 0, 0), -1)
            if data[i, 0] != None and 1 in data[max(0, int(i-show_frame)): i+1][:, 4]:
                img = add_text_to_frame(img, "Start", position=(800, 100), color=(0, 255, 0), font_scale=2)
            if data[i, 0] != None and 1 in data[max(0, int(i-show_frame)): i+1][:, 5]:
                img = add_text_to_frame(img, "End", position=(800, 100), color=(0, 255, 0), font_scale=2)

            PIL_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            PIL_image = Image.fromarray(PIL_image)
            if data[i, 0]!=None:
                bbox = (data[i, 0] - 6, data[i, 1] - 6, data[i, 0] + 6, data[i, 1] + 6)
                draw = ImageDraw.Draw(PIL_image)
                draw.ellipse(bbox, outline='red')
                del draw
            opencvImage = cv2.cvtColor(np.array(PIL_image), cv2.COLOR_RGB2BGR)
            out.write(opencvImage)
            # plt.imshow(opencvImage[:,:,::-1])
            # plt.show()
        # draw.ellipse(bbox, outline ='yellow')
        # glob.glob()
        # 释放视频编写器和销毁所有窗口
        out.release()
        # cv2.destroyAllWindows()