import glob

import torch
from utils.utils import calculate_velocity, add_csv_col, jud_dir, add_text_to_video, interpolation
from utils.feat_utils import get_lag_feature, get_single_lag_feature

from LoadBatches import TSCDataset
from torch.utils.data import DataLoader
from classifiers.resnet import Classifier_RESNET_pt
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

train_tennis_dt = TSCDataset()
test_tennis_dt = TSCDataset(train=False)
epoch_num = 10
device = 'cuda'
target_name = 'first_hit'

net = Classifier_RESNET_pt(2, 2).to(device)
data_loader = DataLoader(train_tennis_dt, batch_size=4, shuffle=True, num_workers=0)
test_data_loader = DataLoader(test_tennis_dt, batch_size=4, shuffle=False, num_workers=0)
# bce_loss = nn.BCELoss(size_average=True)
ce_loss = nn.CrossEntropyLoss(size_average=True)
optimizer = optim.Adam(net.parameters())
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, min_lr=0.0001)

for epoch in range(epoch_num):
    pbar = tqdm(data_loader,
                total=len(data_loader),
                desc=f'Epoch {epoch}')
    for step, batch in enumerate(data_loader):
        inp, label = batch
        inp = inp.to(torch.float32).to(device)
        label = label.to(torch.long).to(device)
        inp = inp.permute(0, 2, 1)
        out = net(inp)
        # loss = bce_loss(out, label)
        loss = ce_loss(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

x_test = torch.from_numpy(test_tennis_dt.x).to(torch.float32).to(device).permute(0, 2, 1)
y_test = (test_tennis_dt.y).astype(np.int)
y_pred = net(x_test).cpu().detach().numpy()
y_pred = np.argmax(y_pred, axis=1)
from utils.metrics import classify_metrics
TP, ALL_HAS, FP, diff = classify_metrics(y_pred, y_test)
print(f'{target_name}_test结果')
print('模型预测正确平均绝对差: ', diff / TP)
print(f'模型预测正确个数/GT个数: {TP}/{ALL_HAS}')
print('没有却检测出来个数: ', FP)



# for step, batch in enumerate(test_data_loader):

