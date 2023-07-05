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


from TrackNet_Three_Frames_Input.utils.utils import calculate_velocity, add_csv_col, jud_dir, add_text_to_video
import csv
import numpy as np


# 读取csv文件
with open('./TrackNet_Three_Frames_Input/gt.csv', newline='', encoding='gbk') as csvfile:
    reader = csv.reader(csvfile)
    next(reader) # 跳过第一行
    data = []
    for row in reader:
        if len(row)==1:
            continue
        else:
            x, y = float(row[1]), float(row[2])
            bounce = int(row[4])
        data.append([x, y, bounce])

# 将数据转换为numpy数组
data = np.array(data)
velocities = []
velocities.append((0, 0))
for i in range(1, data.shape[0]):
    velocities.append((data[i][0]-data[i-1][0], data[i][1]-data[i-1][1]))

velocities = np.array(velocities)
sum_velo = pow(pow(velocities[:, 0], 2)+pow(velocities[:, 1], 2), 0.5)

acceleration = []
acceleration.append((0, 0))
for i in range(1, len(velocities)):
    acc = (velocities[i] - velocities[i-1])
    acceleration.append(acc)
acceleration = np.array(acceleration)
sum_accel = pow(pow(acceleration[:, 0], 2)+pow(acceleration[:, 1], 2), 0.5)
from matplotlib import pyplot as plt
# 绘制曲线图
# plt.plot(acceleration[:, 1], color='black')
plt.plot(data[:, 1], color='black')

bounce = data[:, 2].astype(np.int)
bounce = np.array(np.where(bounce==1))
# 在指定帧上绘制黑点
# plt.scatter(bounce[0], acceleration[bounce][0][:, 1], color='red')
plt.scatter(bounce[0], data[bounce][0][:, 1], color='red')

# 设置x轴和y轴标签
plt.xlabel('frame')
plt.ylabel('y pos')

# 显示图形
plt.show()




