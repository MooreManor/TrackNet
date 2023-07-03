from LoadBatches import TennisDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from Models.TrackNet import TrackNet_pt
import argparse
import torch.nn.functional as F
import os
import cv2
from torchvision.utils import make_grid
import torch.nn as nn
import numpy as np


# --save_weights_path=weights/model --training_images_name="eval.csv" --n_classes=256 --input_height=360 --input_width=640 --batch_size=2
# --save_weights_path=weights/model --training_images_name="eval.csv" --n_classes=256 --input_height=540 --input_width=960 --batch_size=2
#parse parameters
parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path", type = str  )
parser.add_argument("--training_images_name", type = str  )
parser.add_argument("--n_classes", type=int )
parser.add_argument("--input_height", type=int , default = 360  )
parser.add_argument("--input_width", type=int , default = 640 )
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
load_weights = args.load_weights
# step_per_epochs = args.step_per_epochs
step_per_epochs = 200
log_frep = 50
eval_freq = 1
output_width = 1920
output_height = 1080

device = 'cuda'

eval_dt = TennisDataset(images_path=training_images_name, n_classes=n_classes, input_height=input_height, input_width=input_width,
                          # output_height=input_height, output_width=input_width, num_images=64)
                          output_height=input_height, output_width=input_width, train=False)

net = TrackNet_pt(n_classes=n_classes, input_height=input_height, input_width=input_width).to(device)
# data_loader = DataLoader(eval_dt, batch_size=train_batch_size, shuffle=True, num_workers=8)
data_loader = DataLoader(eval_dt, batch_size=train_batch_size, shuffle=False, num_workers=0)

pbar = tqdm(data_loader,
                # total=len(data_loader),
                total=len(data_loader) if step_per_epochs > len(data_loader) else step_per_epochs)

TP = 0
TN = 0
FP = 0
FN = 0
ALL_HAS = 0
for step, batch in enumerate(pbar):
    input, label, vis_output = batch
    input = input.to(device)
    with torch.no_grad():
        pred = net(input)
    pred = pred.reshape((train_batch_size, input_height, input_width))
    # pr = np.array([cv2.resize(img, (output_width, output_height)) for img in pr])
    heatmap = pred.cpu().numpy().astype(np.uint8)
    heatmap = np.array([cv2.resize(img, (output_width, output_height)) for img in heatmap])
    # heatmap = np.array([cv2.resize(img, (output_width, output_height)) for img in heatmap])
    # cv2.resize(pr, (output_width, output_height))
    heatmap = [cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1] for img in heatmap]
    pred_has_circle = [np.max(img) > 0 for img in heatmap]
    pred_circles = [cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=1,minDist=10,param2=2,minRadius=2,maxRadius=7) for img in heatmap]

    outputs = (label*255).reshape((train_batch_size, output_height, output_width))
    # outputs = outputs.reshape((train_batch_size, height, width, n_classes)).argmax(axis=3)
    # outputs = cv2.threshold(outputs, 127, 255, cv2.THRESH_BINARY)
    outputs = outputs.numpy().astype(np.uint8)
    # outputs = [cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1] for img in outputs]
    gt_has_circle = [np.max(img)>0 for img in outputs]
    gt_circles = np.array([np.where(img==np.max(img)) for img in outputs]).squeeze()
    for j in range(train_batch_size):
        tmp_hp = pred_has_circle[j]
        tmp_hg = gt_has_circle[j]
        if tmp_hg==1:
            ALL_HAS += 1
        if pred_circles[j] is None:
            if tmp_hg==0:
                TN += 1
            else:
                FN += 1
            continue
        tmp_pred = pred_circles[j][0][0][:2]
        tmp_gt = gt_circles[j][0][0][:2]

        # if tmp_hp==tmp_hg and tmp_hg==1:
        if tmp_hp==1:
            delta_circle = abs(tmp_pred - tmp_gt)
            if delta_circle[0]*delta_circle[0]+delta_circle[1]*delta_circle[1] < 100 and tmp_hp==tmp_hg:
                TP += 1
            else:
                FP += 1







# import argparse
# import Models , LoadBatches
# from keras import optimizers
# # from tensorflow import keras
# # from keras.utils import plot_model
# from keras.utils.vis_utils import plot_model
# import tensorflow as tf
# # tf.compat.v1.disable_v2_behavior()
# import numpy as np
# from torchvision.utils import make_grid
# import os
# import cv2
# from keras import backend as K
# import math
# from tqdm import tqdm
# from LoadBatches import TennisDataset
# from torch.utils.data import DataLoader
#
#
# width, height = 640, 360
# output_width, output_height = 1280, 720
# parser = argparse.ArgumentParser()
# parser.add_argument("--save_weights_path", type = str  )
# parser.add_argument("--training_images_name", type = str  )
# parser.add_argument("--n_classes", type=int )
# parser.add_argument("--input_height", type=int , default = 360  )
# parser.add_argument("--input_width", type=int , default = 640 )
# parser.add_argument("--epochs", type = int, default = 1000 )
# parser.add_argument("--batch_size", type = int, default = 2 )
# parser.add_argument("--load_weights", type = str , default = "-1")
# parser.add_argument("--step_per_epochs", type = int, default = 200 )
#
# args = parser.parse_args()
#
# training_images_name = args.training_images_name
# train_batch_size = args.batch_size
# n_classes = args.n_classes
# input_height = args.input_height
# input_width = args.input_width
# save_weights_path = args.save_weights_path
# epochs = args.epochs
# load_weights = args.load_weights
# step_per_epochs = args.step_per_epochs
# data_num = 19645
# # iter_num = data_num//train_batch_size
# iter_num = int(math.ceil(data_num // train_batch_size))
#
# modelTN = Models.TrackNet.TrackNet
# m = modelTN(n_classes, input_height=input_height, input_width=input_width)
# m.compile(loss='categorical_crossentropy', optimizer=  'adadelta' , metrics=['accuracy'])
# m.load_weights(  save_weights_path  )
# model_output_height = m.outputHeight
# model_output_width = m.outputWidth
#
# #creat input data and output data
# # Generator = LoadBatches.InputOutputGenerator( training_images_name,  train_batch_size,  n_classes , input_height , input_width , model_output_height , model_output_width)
# Generator = LoadBatches.InputOutputGenerator( training_images_name,  train_batch_size,  n_classes , input_height , input_width , output_height , output_width)
# # input, output = LoadBatches.InputOutputAllData( training_images_name,  train_batch_size,  n_classes , input_height , input_width , output_height , output_width)
# # tennis_dt = TennisDataset(images_path=training_images_name, n_classes=n_classes, input_height=input_height, input_width=input_width,
# #                           # output_height=input_height, output_width=input_width, num_images=64)
# #                           # output_height=input_height, output_width=input_width)
# #                           output_height=output_height, output_width=output_width)
#
# # data_loader = DataLoader(tennis_dt, batch_size=1, shuffle=True, num_workers=8)
# TP = 0
# TN = 0
# FP = 0
# FN = 0
# ALL_HAS = 0
# # pbar = tqdm(data_loader,
# #                 # total=len(data_loader),
# #                 total=iter_num)
# # for step, batch in enumerate(pbar):
# for i in tqdm(range(iter_num)):
# # for i in tqdm(range(2)):
#     print('Step', i)
#     # print('Step', step)
#     # if i==0:
#     #     for k in range(22):
#     #         inputs, outputs = next(Generator)
#     inputs, outputs = next(Generator)
#
#     inputs, outputs = next(Generator)
#     pr = m.predict(inputs)
#     # pr_max = pr.reshape((train_batch_size, height, width, n_classes)).max(axis=3)
#     # pr_max = np.array([cv2.resize(img, (output_width, output_height)) for img in pr_max])
#     # pr_max = cv2.resize(pr_max, (output_width, output_height))
#     # heatmap = cv2.resize(pr, (output_width, output_height))
#     pr = pr.reshape((train_batch_size, height, width, n_classes)).argmax(axis=3)
#     # pr = np.array([cv2.resize(img, (output_width, output_height)) for img in pr])
#     pr = pr.astype(np.uint8)
#     heatmap = np.array([cv2.resize(img, (output_width, output_height)) for img in pr])
#     # cv2.resize(pr, (output_width, output_height))
#     heatmap = [cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1] for img in heatmap]
#     pred_has_circle = [np.max(img) > 0 for img in heatmap]
#     pred_circles = [cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2, minRadius=2, maxRadius=7) for img in heatmap]
#
#     outputs = outputs.reshape((train_batch_size, output_height, output_width, n_classes)).argmax(axis=3)
#     # outputs = outputs.reshape((train_batch_size, height, width, n_classes)).argmax(axis=3)
#     # outputs = cv2.threshold(outputs, 127, 255, cv2.THRESH_BINARY)
#     outputs = outputs.astype(np.uint8)
#     outputs = [cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1] for img in outputs]
#     gt_has_circle = [np.max(img)>0 for img in outputs]
#     gt_circles = [cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2, minRadius=2, maxRadius=7) for img in outputs]
#
#     for j in range(train_batch_size):
#         tmp_hp = pred_has_circle[j]
#         tmp_hg = gt_has_circle[j]
#         if tmp_hg==1:
#             ALL_HAS += 1
#         if pred_circles[j] is None:
#             if tmp_hg==0:
#                 TN += 1
#             else:
#                 FN += 1
#             continue
#         tmp_pred = pred_circles[j][0][0][:2]
#         tmp_gt = gt_circles[j][0][0][:2]
#
#         # if tmp_hp==tmp_hg and tmp_hg==1:
#         if tmp_hp==1:
#             delta_circle = abs(tmp_pred - tmp_gt)
#             if delta_circle[0]*delta_circle[0]+delta_circle[1]*delta_circle[1] < 200 and tmp_hp==tmp_hg:
#                 TP += 1
#             else:
#                 FP += 1
#
#     print("TP:", TP)
#     print("FP:", FP)
#     print("ALL_HAS:", ALL_HAS)
#     print("Precision:", TP/(TP+FP))
#     print("Recall:", TP/ALL_HAS)
#     print('----------------------------------------------------------------')
#         # if tmp_hp==1 and










