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
from sktime.datatypes._panel._convert import from_2d_array_to_nested
from sktime.classification.interval_based import TimeSeriesForestClassifier


# # 读取csv文件
# with open('./TrackNet_Three_Frames_Input/gt.csv', newline='', encoding='gbk') as csvfile:
#     reader = csv.reader(csvfile)
#     next(reader) # 跳过第一行
#     data = []
#     for row in reader:
#         if len(row)==1:
#             continue
#         else:
#             x, y = float(row[1]), float(row[2])
#             bounce = int(row[4])
#         data.append([x, y, bounce])
#
# # 将数据转换为numpy数组
# data = np.array(data)
# velocities = []
# velocities.append((0, 0))
# for i in range(1, data.shape[0]):
#     velocities.append((data[i][0]-data[i-1][0], data[i][1]-data[i-1][1]))
#
# velocities = np.array(velocities)
# sum_velo = pow(pow(velocities[:, 0], 2)+pow(velocities[:, 1], 2), 0.5)
#
# acceleration = []
# acceleration.append((0, 0))
# for i in range(1, len(velocities)):
#     acc = (velocities[i] - velocities[i-1])
#     acceleration.append(acc)
# acceleration = np.array(acceleration)
# sum_accel = pow(pow(acceleration[:, 0], 2)+pow(acceleration[:, 1], 2), 0.5)
# from matplotlib import pyplot as plt
# # 绘制曲线图
# # plt.plot(acceleration[:, 1], color='black')
# plt.plot(data[:, 1], color='black')
#
# bounce = data[:, 2].astype(np.int)
# bounce = np.array(np.where(bounce==1))
# # 在指定帧上绘制黑点
# # plt.scatter(bounce[0], acceleration[bounce][0][:, 1], color='red')
# plt.scatter(bounce[0], data[bounce][0][:, 1], color='red')
#
# # 设置x轴和y轴标签
# plt.xlabel('frame')
# plt.ylabel('y pos')
#
# # 显示图形
# plt.show()
def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """
    return np.isnan(y), lambda z: z.nonzero()[0]

def interpolation(coords):
  coords =coords.copy()
  x, y = [x[0] if x is not None else np.nan for x in coords], [x[1] if x is not None else np.nan for x in coords]

  xxx = np.array(x) # x coords
  yyy = np.array(y) # y coords

  nons, yy = nan_helper(xxx)
  xxx[nons]= np.interp(yy(nons), yy(~nons), xxx[~nons])
  nans, xx = nan_helper(yyy)
  yyy[nans]= np.interp(xx(nans), xx(~nans), yyy[~nans])

  newCoords = [*zip(xxx,yyy)]

  return newCoords

import pandas as pd
with open('./TrackNet_Three_Frames_Input/data/csv/BigDataFrame.csv', newline='', encoding='gbk') as csvfile:
    reader = csv.reader(csvfile)
    next(reader) # 跳过第一行
    data = []
    for row in reader:
        if len(row)==1:
            continue
        else:
            x, y = float(row[0]), float(row[1])
            v = float(row[2])
            bounce = int(row[3])
        data.append([x, y, v, bounce])

from sklearn import tree
bounce = list(np.array(data)[:, 3])
bounce = np.array([int(x) for x in bounce])
x = list(np.array(data)[:, 0])
y = list(np.array(data)[:, 1])
v = list(np.array(data)[:, 2])
test_df = pd.DataFrame({'x': x, 'y':y, 'V': v})
for i in range(20, 0, -1):
    test_df[f'lagX_{i}'] = test_df['x'].shift(i, fill_value=0)
for i in range(20, 0, -1):
    test_df[f'lagY_{i}'] = test_df['y'].shift(i, fill_value=0)
for i in range(20, 0, -1):
    test_df[f'lagV_{i}'] = test_df['V'].shift(i, fill_value=0)
test_df.drop(['x', 'y', 'V'], 1, inplace=True)
Xs = test_df[['lagX_20', 'lagX_19', 'lagX_18', 'lagX_17', 'lagX_16',
        'lagX_15', 'lagX_14', 'lagX_13', 'lagX_12', 'lagX_11', 'lagX_10',
        'lagX_9', 'lagX_8', 'lagX_7', 'lagX_6', 'lagX_5', 'lagX_4', 'lagX_3',
        'lagX_2', 'lagX_1']]
Xs = from_2d_array_to_nested(Xs.to_numpy())

Ys = test_df[['lagY_20', 'lagY_19', 'lagY_18', 'lagY_17',
        'lagY_16', 'lagY_15', 'lagY_14', 'lagY_13', 'lagY_12', 'lagY_11',
        'lagY_10', 'lagY_9', 'lagY_8', 'lagY_7', 'lagY_6', 'lagY_5', 'lagY_4',
        'lagY_3', 'lagY_2', 'lagY_1']]
Ys = from_2d_array_to_nested(Ys.to_numpy())

Vs = test_df[['lagV_20', 'lagV_19', 'lagV_18',
    'lagV_17', 'lagV_16', 'lagV_15', 'lagV_14', 'lagV_13', 'lagV_12',
    'lagV_11', 'lagV_10', 'lagV_9', 'lagV_8', 'lagV_7', 'lagV_6', 'lagV_5',
    'lagV_4', 'lagV_3', 'lagV_2', 'lagV_1']]
Vs = from_2d_array_to_nested(Vs.to_numpy())

# X = pd.concat([Xs, Ys, Vs], 1)
X = pd.concat([Vs], 1)
# X = [[0, 0], [1, 1]]
# Y = [0, 1]
Y = bounce
# clf = tree.DecisionTreeClassifier()
clf = TimeSeriesForestClassifier()
clf.fit(X, Y)

dir_path = '/datasetb/tennis/'
# games = ['game7/Clip1', 'game2/Clip1', 'game3/Clip1', 'game4/Clip1', 'game5/Clip1', 'game6/Clip1', 'game7/Clip1', 'game8/Clip1', 'game9/Clip1', 'game10/Clip1']
games = ['game7/Clip4']
for game in games:
    csv_path = dir_path+game+'/Label.csv'
    with open(csv_path, newline='', encoding='gbk') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # 跳过第一行
        X_test = []
        Y_test = []
        for row in reader:
            if row[3]=='':
                # x, y = None, None
                # bounce = 0
                X_test.append(None)
                Y_test.append(0)
            else:
                x, y = float(row[2]), float(row[3])
                # v = float(row[2])
                bounce = 1 if int(row[4])==2 else 0
                X_test.append([x, y])
                Y_test.append(bounce)
X_test = interpolation(X_test)
Vx = [0]
Vy = [0]
V = []
for i in range(len(X_test)-1):
  p1 = X_test[i]
  p2 = X_test[i+1]
  x = (p1[0]-p2[0])
  y = (p1[1]-p2[1])
  Vx.append(x)
  Vy.append(y)

for i in range(len(Vx)):
  vx = Vx[i]
  vy = Vy[i]
  v = (vx**2+vy**2)**0.5
  V.append(v)
from sklearn.metrics import precision_score, recall_score

X_test = np.array(X_test)
V = np.array(V)
V = V.reshape((-1, 1))
X_test = np.concatenate((X_test, V), axis=1)
x = list(np.array(X_test)[:, 0])
y = list(np.array(X_test)[:, 1])
v = list(np.array(X_test)[:, 2])
test_df = pd.DataFrame({'x': x, 'y':y, 'V': v})
for i in range(20, 0, -1):
    test_df[f'lagX_{i}'] = test_df['x'].shift(i, fill_value=0)
for i in range(20, 0, -1):
    test_df[f'lagY_{i}'] = test_df['y'].shift(i, fill_value=0)
for i in range(20, 0, -1):
    test_df[f'lagV_{i}'] = test_df['V'].shift(i, fill_value=0)
test_df.drop(['x', 'y', 'V'], 1, inplace=True)
Xs = test_df[['lagX_20', 'lagX_19', 'lagX_18', 'lagX_17', 'lagX_16',
        'lagX_15', 'lagX_14', 'lagX_13', 'lagX_12', 'lagX_11', 'lagX_10',
        'lagX_9', 'lagX_8', 'lagX_7', 'lagX_6', 'lagX_5', 'lagX_4', 'lagX_3',
        'lagX_2', 'lagX_1']]
Xs = from_2d_array_to_nested(Xs.to_numpy())

Ys = test_df[['lagY_20', 'lagY_19', 'lagY_18', 'lagY_17',
        'lagY_16', 'lagY_15', 'lagY_14', 'lagY_13', 'lagY_12', 'lagY_11',
        'lagY_10', 'lagY_9', 'lagY_8', 'lagY_7', 'lagY_6', 'lagY_5', 'lagY_4',
        'lagY_3', 'lagY_2', 'lagY_1']]
Ys = from_2d_array_to_nested(Ys.to_numpy())

Vs = test_df[['lagV_20', 'lagV_19', 'lagV_18',
    'lagV_17', 'lagV_16', 'lagV_15', 'lagV_14', 'lagV_13', 'lagV_12',
    'lagV_11', 'lagV_10', 'lagV_9', 'lagV_8', 'lagV_7', 'lagV_6', 'lagV_5',
    'lagV_4', 'lagV_3', 'lagV_2', 'lagV_1']]
Vs = from_2d_array_to_nested(Vs.to_numpy())

bounce = list(np.array(Y_test))
bounce = np.array([int(x) for x in bounce])
# X = pd.concat([Xs, Ys, Vs], 1)
X_test = pd.concat([Vs], 1)
y_pred = clf.predict(X_test)
precision=precision_score(bounce, y_pred, average='macro')
recall=recall_score(bounce, y_pred, average='macro')
print('precision: ', precision)
print('recall: ', recall)





