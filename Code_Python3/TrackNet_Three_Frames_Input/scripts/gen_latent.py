from LoadBatches import TennisDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from Models.TrackNet import TrackNet_pt
import argparse
import torch.nn.functional as F
import os
import cv2
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
from utils.train_utils import get_log_dir, train_summaries, eval_summaries
import logging


parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path", type = str  )
parser.add_argument("--training_images_name", type = str  )
parser.add_argument("--n_classes", type=int )
parser.add_argument("--input_height", type=int , default = 360  )
parser.add_argument("--input_width", type=int , default = 640 )
parser.add_argument("--epochs", type = int, default = 1000 )
parser.add_argument("--batch_size", type = int, default = 2 )
parser.add_argument("--load_weights", type = str , default = "-1")
# parser.add_argument("--step_per_epochs", type = int, default = 200 )

args = parser.parse_args()
training_images_name = args.training_images_name
train_batch_size = args.batch_size
n_classes = args.n_classes
input_height = args.input_height
input_width = args.input_width
save_weights_path = args.save_weights_path
epochs = args.epochs
load_weights = args.load_weights
# step_per_epochs = args.step_per_epochs
step_per_epochs = 100
train_log_frep = 50
eval_log_frep = 200
eval_freq = 1

device = 'cuda'



tennis_dt = TennisDataset(images_path=training_images_name, n_classes=n_classes, input_height=input_height, input_width=input_width,
                          # output_height=input_height, output_width=input_width, num_images=64)
                          output_height=input_height, output_width=input_width)

# data_loader = DataLoader(tennis_dt, batch_size=train_batch_size, shuffle=True, num_workers=8)
data_loader = DataLoader(tennis_dt, batch_size=train_batch_size, shuffle=False, num_workers=8)
import torch
from Models.uiunet import UIUNET
net = UIUNET(9, 1).to(device)
net.load_state_dict(torch.load('./weights/model.pt.uiu.best'), strict=True)

pbar = tqdm(data_loader,
                total=len(data_loader),
                # total=len(data_loader) if step_per_epochs > len(data_loader) else step_per_epochs,
                desc=f'Epoch 0')
net.eval()
for step, batch in enumerate(pbar):
    # pass
    input, label, vis_output, img_path = batch
    input = input.to(device)
    label = label.to(device)
    batch_size = input.shape[0]
    with torch.no_grad():
        d0, d1, d2, d3, d4, d5, d6, feat = net(input)
        pred = d0
    for i in range(batch_size):
        save_path = ('/'.join(img_path[i].split('/')[:3] + ['npy'] + img_path[i].split('/')[3:]))[:-4]+'.npy'
        dir_path = os.path.dirname(save_path)
        os.makedirs(dir_path, exist_ok=True)
        save_feat = feat[i].cpu().detach().numpy()
        np.save(save_path, save_feat)