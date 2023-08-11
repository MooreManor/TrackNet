from LoadBatches import VideoTSCDataset
from torch.utils.data import DataLoader
from classifiers.resnet import Classifier_RESNET_pt
import torch
from utils.metrics import classify_metrics
import numpy as np


device = 'cuda'
target_name = ''
var_list = ['x', 'y', 'vx', 'vy', 'ax', 'ay', 'v', 'a']
csv_path = '/datasetb/tennis/haluo/csv/16811385934162.csv'
tscdt = VideoTSCDataset(csv_path, var_list=var_list)
var_num = len(var_list)
thr = 0.2

# data_loader = DataLoader(tscdt, batch_size=64, shuffle=True, num_workers=0)
x_test = tscdt.x
y_test = tscdt.y

net = Classifier_RESNET_pt(2, var_num).to(device)
net.load_state_dict(torch.load('./weights/model.tsc.resnet'), strict=True)

x_test = torch.from_numpy(x_test).permute(0, 2, 1).to(torch.float32).to(device)
y_pred = net(x_test).cpu().detach().numpy()
y_pred = y_pred[:, 1]
y_pred_cls = np.where(y_pred > thr, 1, 0)

gt_pos = np.where(y_test==1)
pred_pos = np.where(y_pred_cls==0)
inter = np.intersect1d(gt_pos, pred_pos)
debug_cls = y_pred[inter]
# pos_pred_0 = np.where(y_pred==0)
TP, ALL_HAS, FP, diff = classify_metrics(y_pred_cls, y_test)
print(f'{target_name}_test结果')
# print('模型预测正确平均绝对差: ', diff / TP)
print(f'模型预测正确个数/GT个数: {TP}/{ALL_HAS}')
print('没有却检测出来个数: ', FP)

