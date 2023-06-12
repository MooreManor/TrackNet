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
from TrackNet_Three_Frames_Input.utils.utils import calculate_velocity, add_csv_col, jud_dir, add_text_to_video
import csv
import numpy as np

# 读取csv文件
with open('tennis.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader) # 跳过第一行
    data = []
    for row in reader:
        if row[1]=='None':
            x, y = None, None
        else:
            x, y = float(row[1]), float(row[2])
        data.append([x, y])

# 将数据转换为numpy数组
data = np.array(data)
vols = calculate_velocity(data)
y_vol = vols[:, 1]
hit, start = jud_dir(y_vol)
add_text_to_video('./output3_TrackNet.mp4', 'hit.mp4', hit, start)
#
# vols = calculate_velocity(data)
# add_csv_col(folder='./', new_col_name='x速度', data=vols[:, 0], file_name='tennis_loc.csv')
# add_csv_col(folder='./', new_col_name='y速度', data=vols[:, 1], file_name='tennis_loc.csv')

import cv2
import numpy as np

# 读取两个图像帧
# frame1 = cv2.imread('000002.png')
# frame2 = cv2.imread('000003.png')
#
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