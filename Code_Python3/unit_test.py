# from TrackNet_Three_Frames_Input.LoadBatches import TennisDataset
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import torch
# from TrackNet_Three_Frames_Input.Models.TrackNet import TrackNet_pt
# device = 'cuda'
# def pt_categorical_crossentropy(pred, label):
#     """
#     使用pytorch 来实现 categorical_crossentropy
#     """
#     # print(-label * torch.log(pred))
#     return torch.sum(-label * torch.log(pred))
#
# #
# tennis_dt = TennisDataset(images_path="TrackNet_Three_Frames_Input/training_model3_mine.csv", n_classes=256, input_height=360, input_width=640,
#                           output_height=360, output_width=640)
#
# data_loader = DataLoader(tennis_dt, batch_size=2, shuffle=False, num_workers=0)
# net = TrackNet_pt(n_classes=256, input_height=360, input_width=640).to(device)
# optimizer = torch.optim.Adam(net.parameters(), lr=1.0)
#
# pbar = tqdm(data_loader,
#                  total=len(data_loader),
#                  desc='Train',)
# epochs = 100
# for epoch in range(epochs):
#     for step, batch in enumerate(pbar):
#         # pbar.set_description(f"No.{step}")
#         input, label = batch
#         input = input.to(device)
#         label = label.to(device)
#         pred = net(input)
#         loss = pt_categorical_crossentropy(pred, label)
#         pbar.set_postfix({"loss": float(loss.cpu().detach().numpy())})
#         loss.backward()
#         optimizer.step()


# loss
# #--------------------
# import tensorflow as tf
# import numpy as np
# # y_true = [[[0., 0.9, 1.0, 0.0, 1.0, 0.0]]]
# # y_pred = [[[0.4, 0.6, 0.8, 0.2, 0.9, 0.2]]]
#
# y_true = [[[0.,0.9]]]
# y_pred = [[[0.4,0.6]]]
# loss = tf.keras.losses.categorical_crossentropy(np.array(y_true), np.array(y_pred))
# loss.numpy()
# print('tf.keras.losses.categorical_crossentropy', loss)
#
# loss = -(0 * tf.math.log(0.4) + 0.9 * tf.math.log(0.6))
# print('tf.math.log', loss)
#
# import torch
#
# loss = -(0 * torch.log(torch.tensor(0.4)) + 0.9 * torch.log(torch.tensor(0.6)))
# print('torch.loss', loss)
#
#
# def pt_categorical_crossentropy(pred, label):
#     """
#     使用pytorch 来实现 categorical_crossentropy
#     """
#     # print(-label * torch.log(pred))
#     return torch.sum(-label * torch.log(pred))
#
#
# loss = pt_categorical_crossentropy(torch.tensor(y_pred), torch.tensor(y_true))
# print(loss)


# test
#------------------------------
from TrackNet_Three_Frames_Input.utils.utils import calculate_velocity, add_csv_col, hsv_thr_vis
# import csv
# import numpy as np
#
# # 读取csv文件
# with open('tennis_loc.csv', newline='') as csvfile:
#     reader = csv.reader(csvfile)
#     next(reader) # 跳过第一行
#     data = []
#     for row in reader:
#         if row[1]=='None':
#             x, y = None, None
#         else:
#             x, y = float(row[1]), float(row[2])
#         data.append([x, y])
#
# # 将数据转换为numpy数组
# data = np.array(data)
#
# vols = calculate_velocity(data)
# add_csv_col(folder='./', new_col_name='x速度', data=vols[:, 0], file_name='tennis_loc.csv')
# add_csv_col(folder='./', new_col_name='y速度', data=vols[:, 1], file_name='tennis_loc.csv')

import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob
import os

# # 读取两个图像帧
# frame1 = cv2.imread('000002.png')
# frame2 = cv2.imread('000003.png')

# hsv = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
# plt.imshow(hsv)
# plt.show()
# h, s, v = cv2.split(hsv)
# plt.imshow(h)
# plt.show()
# plt.imshow(s)
# plt.show()
# plt.imshow(v)
# plt.show()
#--------------------------------------------------------------------
# img_dir = 'TrackNet_Three_Frames_Input/tmp/output3/imgs'
# dst_dir = 'TrackNet_Three_Frames_Input/tmp/output3/deb'
# dst_dir1 = 'TrackNet_Three_Frames_Input/tmp/output3/debh'
# dst_dir2 = 'TrackNet_Three_Frames_Input/tmp/output3/debs'
# dst_dir3 = 'TrackNet_Three_Frames_Input/tmp/output3/debv'
# # dst_dir = 'TrackNet_Three_Frames_Input/tmp/output3/deb'
# os.makedirs(dst_dir, exist_ok=True)
# os.makedirs(dst_dir1, exist_ok=True)
# os.makedirs(dst_dir2, exist_ok=True)
# os.makedirs(dst_dir3, exist_ok=True)
# img_list = glob.glob(os.path.join(img_dir, '*.png'))
# dst_list = [dst_dir+'/'+img_path.split('/')[-1] for img_path in img_list]
# dst_list1 = [dst_dir1+'/'+img_path.split('/')[-1] for img_path in img_list]
# dst_list2 = [dst_dir2+'/'+img_path.split('/')[-1] for img_path in img_list]
# dst_list3 = [dst_dir3+'/'+img_path.split('/')[-1] for img_path in img_list]
# for i, img_path in enumerate(img_list):
#     img = cv2.imread(img_path)
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     h, s, v = cv2.split(hsv)
#     cv2.imwrite(dst_list[i], hsv)
#     cv2.imwrite(dst_list1[i], h)
#     cv2.imwrite(dst_list2[i], s)
#     cv2.imwrite(dst_list3[i], v)

# -----------------------------------------------------------------------------
# 回调函数
# def nothing(*arg):
#     pass
# def Trackbar_Init():
#     # 1 create windows
#     cv2.namedWindow('h_binary')
#     cv2.namedWindow('s_binary')
#     cv2.namedWindow('v_binary')
#     # 2 Create Trackbar
#     cv2.createTrackbar('hmin', 'h_binary', 6, 179, nothing)
#     cv2.createTrackbar('hmax', 'h_binary', 26, 179, nothing)
#     cv2.createTrackbar('smin', 's_binary', 110, 255, nothing)
#     cv2.createTrackbar('smax', 's_binary', 255, 255, nothing)
#     cv2.createTrackbar('vmin', 'v_binary', 140, 255, nothing)
#     cv2.createTrackbar('vmax', 'v_binary', 255, 255, nothing)
#     #   创建滑动条     滑动条值名称 窗口名称   滑动条值 滑动条阈值 回调函数
# Trackbar_Init()
# hmin = cv2.getTrackbarPos('hmin', 'h_binary')
# hmax = cv2.getTrackbarPos('hmax', 'h_binary')
# smin = cv2.getTrackbarPos('smin', 's_binary')
# smax = cv2.getTrackbarPos('smax', 's_binary')
# vmin = cv2.getTrackbarPos('vmin', 'v_binary')
# vmax = cv2.getTrackbarPos('vmax', 'v_binary')
#
# image = cv2.imread('000002.png')
# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# # cv2.imshow('hsv', hsv)
# h, s, v = cv2.split(hsv)
# h_binary = cv2.inRange(np.array(h), np.array(hmin), np.array(hmax))
# s_binary = cv2.inRange(np.array(s), np.array(smin), np.array(smax))
# v_binary = cv2.inRange(np.array(v), np.array(vmin), np.array(vmax))
#
# # 5 Show
# cv2.imshow('h_binary', h_binary)
# cv2.imshow('s_binary', s_binary)
# cv2.imshow('v_binary', v_binary)
# cv2.waitKey(0)
#---------------------------------------------------------------------
import cv2
import numpy as np

def Get_HSV(image):
    # 1 get trackbar's value
    hmin = cv2.getTrackbarPos('hmin', 'h_binary')
    hmax = cv2.getTrackbarPos('hmax', 'h_binary')
    smin = cv2.getTrackbarPos('smin', 's_binary')
    smax = cv2.getTrackbarPos('smax', 's_binary')
    vmin = cv2.getTrackbarPos('vmin', 'v_binary')
    vmax = cv2.getTrackbarPos('vmax', 'v_binary')

    # 2 to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imshow('hsv', hsv)
    h, s, v = cv2.split(hsv)

    # 3 set threshold (binary image)
    # if value in (min, max):white; otherwise:black
    h_binary = cv2.inRange(np.array(h), np.array(hmin), np.array(hmax))
    s_binary = cv2.inRange(np.array(s), np.array(smin), np.array(smax))
    v_binary = cv2.inRange(np.array(v), np.array(vmin), np.array(vmax))

    # 4 get binary（对H、S、V三个通道分别与操作）
    binary = cv2.bitwise_and(h_binary, cv2.bitwise_and(s_binary, v_binary))

    # 5 Show current threshold values on image
    font = cv2.FONT_HERSHEY_SIMPLEX
    # cv2.putText(binary, f'hmin:{hmin} hmax:{hmax}', (10, 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    # cv2.putText(binary, f'smin:{smin} smax:{smax}', (10, 40), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    # cv2.putText(binary, f'vmin:{vmin} vmax:{vmax}', (10, 60), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # 6 Show
    cv2.imshow('h_binary', h_binary)
    cv2.imshow('s_binary', s_binary)
    cv2.imshow('v_binary', v_binary)
    # cv2.imshow('binary', binary)

    return binary


# # create trackbars
# cv2.namedWindow('h_binary')
# cv2.createTrackbar('hmin', 'h_binary', 0, 255, lambda x: None)
# cv2.createTrackbar('hmax', 'h_binary', 255, 255, lambda x: None)
#
# cv2.namedWindow('s_binary')
# cv2.createTrackbar('smin', 's_binary', 0, 255, lambda x: None)
# cv2.createTrackbar('smax', 's_binary', 255, 255, lambda x: None)
#
# cv2.namedWindow('v_binary')
# cv2.createTrackbar('vmin', 'v_binary', 0, 255, lambda x: None)
# cv2.createTrackbar('vmax', 'v_binary', 255, 255, lambda x: None)
#
#
# # load image
# image = cv2.imread('000002.png')
#
# while True:
#     binary = Get_HSV(image)
#     key = cv2.waitKey(1) & 0xFF
#     if key == 27:  # press 'ESC' to exit
#         break
#
# cv2.destroyAllWindows()
hsv_thr_vis('000002.png')
# # 转换为灰度图像
# gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
# gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
#
# # 使用Lucas-Kanade光流算法计算光流场
# flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
#
# # 可视化光流场
# flow_x = flow[..., 0]
# flow_y = flow[..., 1]
#
# # 创建画布
# h, w = gray1.shape
# canvas = np.zeros((h, w, 3), dtype=np.uint8)
#
# # 画箭头表示光流向量
# step = 10  # 控制箭头密度
# for y in range(0, h, step):
#     for x in range(0, w, step):
#         dx = int(flow_x[y, x])
#         dy = int(flow_y[y, x])
#         cv2.arrowedLine(canvas, (x, y), (x + dx, y + dy), (0, 255, 0), 1, tipLength=0.5)
#
# # 显示结果
# cv2.imshow('Optical Flow', canvas)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
# flow_val =  np.linalg.norm(flow, axis=-1)
# max_pos = np.argmax(flow_val)
# y, x = np.unravel_index(max_pos, flow_val.shape)
# pass