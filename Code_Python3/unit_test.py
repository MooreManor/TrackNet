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
# from TrackNet_Three_Frames_Input.utils.utils import calculate_velocity, add_csv_col
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

import numpy as np
from scipy.linalg import inv

# 系统动态模型（状态转移矩阵）
A = np.array([[1, 1], [0, 1]])

# 测量模型（测量矩阵）
H = np.array([[1, 0]])

# 状态协方差矩阵（初始状态不确定性）
P = np.array([[1000, 0], [0, 1000]])

# 过程噪声协方差矩阵
Q = np.array([[1, 0], [0, 1]])

# 测量噪声协方差矩阵
R =np.array([[1]])

# 初始状态
x = np.array([[0], [0]])

# 测量数据
measurements = [1, 2, 3, 4, 5]

for z in measurements:
    # 执行卡尔曼滤波
    x_pred = np.dot(A, x)     # 预测状态
    P_pred = np.dot(np.dot(A, P), A.T) + Q   # 预测协方差矩阵
    K = np.dot(np.dot(P_pred, H.T), inv(np.dot(np.dot(H, P_pred), H.T) + R))   # 卡尔曼增益
    x = x_pred + np.dot(K, (z - np.dot(H, x_pred)))    # 更新状态
    P = np.dot((np.eye(2) - np.dot(K, H)), P_pred)     # 更新协方差矩阵

    # 输出估计值
    print("估计值:", x.flatten())