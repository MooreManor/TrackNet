import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import warnings

# 忽略所有警告
warnings.filterwarnings('ignore')

# 忽略特定类型的警告
warnings.filterwarnings('ignore', category=DeprecationWarning)
from utils.utils import calculate_velocity, add_csv_col, jud_dir, add_text_to_video, interpolation, jud_y_dir
from utils.metrics import classify_metrics
from utils.feat_utils import get_lag_feature
import csv
import numpy as np
from sktime.datatypes._panel._convert import from_2d_array_to_nested
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.classification.kernel_based import RocketClassifier
from sktime.classification.hybrid import HIVECOTEV2
from sktime.classification.interval_based import DrCIF
from sktime.classification.compose import ColumnEnsembleClassifier
from sktime.transformations.panel.compose import ColumnConcatenator
from sklearn import tree


import pandas as pd
import glob

# X = np.empty((0, 20, 3))
X = np.empty((0, 60))
# X_test = np.empty((0, 20, 3))
X_test = np.empty((0, 60))
Y = np.empty((0,))
Y_test = np.empty((0,))
res_TP = 0
res_ALL_HAS = 0
res_FP = 0
res_diff = 0

test_game = ['game9', 'game10']
csv_file_all = glob.glob('/datasetb/tennis/' + '/**/Label.csv', recursive=True)
for csv_path in csv_file_all:
    data = []
    test = 0
    if any(x in csv_path for x in test_game):
        test = 1
    else:
        continue

    with open(csv_path, newline='', encoding='gbk') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # 跳过第一行
        for row in reader:
            if row[2] == '':
                data.append([None, None, 0])
            else:
                x, y = float(row[2]), float(row[3])
                if int(row[4])==1:
                    bounce = 1
                else:
                    bounce = 0
                data.append([x, y, bounce])
    bounce = list(np.array(data)[:, 2])
    bounce = np.array([int(x) for x in bounce])
    xy = [l[:2] for l in data]
    xy = interpolation(xy)
    x = list(np.array(xy)[:, 0])
    y = list(np.array(xy)[:, 1])
    v = np.diff(xy, axis=0)
    v = np.pad(v, ((1, 0), (0, 0)), 'constant', constant_values=0)
    vx = v[:, 0]
    vy = v[:, 1]
    v = pow(pow(v[:, 0], 2) + pow(v[:, 1], 2), 0.5)

    hit, start, end = jud_y_dir(vy)
    TP, ALL_HAS, FP, diff = classify_metrics(hit, bounce)
    res_TP += TP
    res_ALL_HAS += ALL_HAS
    res_FP += FP
    res_diff += diff

print('决策树结果')
print('模型预测正确平均绝对差: ', res_diff/res_TP)
print(f'模型预测正确个数/GT个数: {res_TP}/{res_ALL_HAS}')
print('没有开局却检测出来开局个数: ', res_FP)