import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from utils.utils import calculate_velocity, add_csv_col, jud_dir, add_text_to_video, interpolation
from utils.metrics import classify_metrics
import csv
import numpy as np
from sktime.datatypes._panel._convert import from_2d_array_to_nested
from sktime.classification.interval_based import TimeSeriesForestClassifier

import pandas as pd
with open('./data/csv/BigDataFrame.csv', newline='', encoding='gbk') as csvfile:
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

X = pd.concat([Xs, Ys, Vs], 1)
# X = pd.concat([Vs], 1)
# X = [[0, 0], [1, 1]]
# Y = [0, 1]
Y = bounce
# clf = tree.DecisionTreeClassifier()
# clf = TimeSeriesForestClassifier()
# clf.fit(X, Y)
from sktime.classification.kernel_based import RocketClassifier
from sktime.classification.hybrid import HIVECOTEV2
from sktime.classification.interval_based import DrCIF
from sktime.classification.compose import ColumnEnsembleClassifier
from sktime.transformations.panel.compose import ColumnConcatenator
# clf = ColumnConcatenator() * DrCIF(n_estimators=10, n_intervals=5)
# clf = HIVECOTEV2(time_limit_in_minutes=0.2)
clf = RocketClassifier(num_kernels=2000)
# clf = tree.DecisionTreeClassifier()
# clf = ColumnEnsembleClassifier(
#     estimators=[
#         ("DrCIF0", DrCIF(n_estimators=10, n_intervals=5), [0]),
#         ("ROCKET3", RocketClassifier(num_kernels=1000), [3]),
#     ]
# )
X = X.values
len_X = len(X)
X = np.stack([np.array(series.values) for series in X.reshape(-1)])
# X = X.reshape(len_X, 3, 20)
X = X.reshape(len_X, 20, 3)
# X = X.reshape(len_X, 60)
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
                x, y = None, None
                # bounce = 0
                # X_test.append(None)
                X_test.append([x, y])
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
X_test = pd.concat([Xs, Ys, Vs], 1)
len_XTest = X_test.shape[0]
X_test = X_test.values
X_test = np.stack([np.array(series.values) for series in X_test.reshape(-1)])
# X_test = X_test.reshape(len_XTest, 3, 20)
X_test = X_test.reshape(len_XTest, 20, 3)
# X_test = X_test.reshape(len_XTest, 60)
y_pred = clf.predict(X_test)
TP, ALL_HAS, FP, diff = classify_metrics(y_pred, bounce)
# precision=precision_score(bounce, y_pred, average='macro')
# recall=recall_score(bounce, y_pred, average='macro')
# print('precision: ', precision)
# print('recall: ', recall)
print('决策树结果')
print('模型预测正确平均绝对差: ', diff/TP)
print(f'模型预测正确个数/GT个数: {TP}/{ALL_HAS}')
print('没有开局却检测出来开局个数: ', FP)
