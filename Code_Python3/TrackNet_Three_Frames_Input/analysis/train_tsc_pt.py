import glob

import torch
from utils.utils import calculate_velocity, add_csv_col, jud_dir, add_text_to_video, interpolation
from utils.feat_utils import get_lag_feature, get_single_lag_feature

from LoadBatches import TSCDataset
from torch.utils.data import DataLoader
from classifiers.resnet import Classifier_RESNET_pt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

target_name = 'first_hit'
# target_name = 'last_hit'
var_list = ['x', 'y', 'vx', 'vy', 'ax', 'ay', 'v', 'a']
# var_list = ['x', 'y']

var_num = len(var_list)
train_tennis_dt = TSCDataset(var_list=var_list, target_name=target_name)
test_tennis_dt = TSCDataset(train=False, var_list=var_list, target_name=target_name)
epoch_num = 200
device = 'cuda'
save_weights_path = 'weights/model'

net = Classifier_RESNET_pt(2, var_num).to(device)
data_loader = DataLoader(train_tennis_dt, batch_size=64, shuffle=True, num_workers=0)
test_data_loader = DataLoader(test_tennis_dt, batch_size=4, shuffle=False, num_workers=0)
bce_loss = nn.BCELoss(size_average=True)
ce_loss = nn.CrossEntropyLoss(size_average=True)
optimizer = optim.Adam(net.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, min_lr=0.0001)


x_test = torch.from_numpy(test_tennis_dt.x).to(torch.float32).to(device).permute(0, 2, 1)
y_test = (test_tennis_dt.y).astype(np.int)

thr_list =[0.1, 0.2, 0.3, 0.4, 0.5]
for epoch in range(epoch_num):
    pbar = tqdm(data_loader,
                total=len(data_loader),
                desc=f'Epoch {epoch}')
    for step, batch in enumerate(pbar):
        inp, label = batch
        label = F.one_hot(label.to(torch.long), num_classes=2).to(torch.float32)
        inp = inp.to(torch.float32).to(device)
        # label = label.to(torch.long).to(device)
        label = label.to(device)
        inp = inp.permute(0, 2, 1)
        out = net(inp)
        loss = bce_loss(out, label)
        # loss = ce_loss(out, label)
        pbar.set_postfix({"loss": float(loss.cpu().detach().numpy())})
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch%10==0 and epoch>=10:
        precision = []
        recall = []
        for thr in thr_list:
            print(f'----------------------Epoch {epoch} | thr {thr}----------------------')
            y_pred = net(x_test).cpu().detach().numpy()
            # y_pred = np.argmax(y_pred, axis=1)
            # y_pred = y_pred[:, 1] > thr
            y_pred = y_pred[:, 1]
            y_pred_cls = np.where(y_pred > thr, 1, 0)

            from utils.metrics import classify_metrics

            TP, ALL_HAS, FP, diff = classify_metrics(y_pred_cls, y_test)
            precision.append(TP/(TP+FP))
            recall.append(TP/ALL_HAS)
            print(f'{target_name}_test结果')
            # print('模型预测正确平均绝对差: ', diff / TP)
            print(f'模型预测正确个数/GT个数: {TP}/{ALL_HAS}')
            print('没有却检测出来个数: ', FP)
        from sklearn.metrics import auc
        from scipy.integrate import trapz
        # auc_score = auc(recall, precision)
        auc_score = trapz(precision, recall[::-1])
        print('AUC score: ', auc_score)

    torch.save(net.state_dict(), save_weights_path + ".tsc.resnet")
    pbar.close()





# for step, batch in enumerate(test_data_loader):

