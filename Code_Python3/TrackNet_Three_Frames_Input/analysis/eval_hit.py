import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import glob
import numpy as np
import cv2
import torch
from Models.uiunet import UIUNET
from utils.kp_utils import get_heatmap_preds
from utils.utils import interpolation, calculate_velocity, jud_y_dir
import csv

device = 'cuda'
csv_path = '/datasetb/tennis/haluo/csv'
vid_path = '/datasetb/tennis/haluo/vids/'
csv_file = glob.glob(f'{csv_path}/*.csv')
vid_file = [vid_path + os.path.basename(x)[:-4]+'.mp4' for x in csv_file]
net = UIUNET(9, 1).to(device)
net.load_state_dict(torch.load('./weights/model.pt.uiu.latest'), strict=True)
train_batch_size = 1
input_height=360
input_width=640
output_width = 1920
output_height = 1080

TP = 0
ALL_HAS = 0
FP = 0
net.eval()
for i, vid in enumerate(vid_file):
    label_path = csv_file[i]
    gt_csv = []
    with open(label_path, 'r', encoding='gbk') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # 跳过第一行
        for row in reader:
            if len(row) == 1:
                x, y, hit = None, None, 0
            else:
                x, y = float(row[1]), float(row[2])
                hit = int(row[3])
            gt_csv.append([x, y, hit])

    gt_csv = np.array(gt_csv)
    gt_hit = gt_csv[:, 2]
    print(f'{os.path.basename(vid)} Start')
    currentFrame = 0

    img, img1, img2 = None, None, None
    video = cv2.VideoCapture(vid)
    width, height = 640, 360

    ret, img1 = video.read()
    currentFrame += 1
    ori_img1 = img1.copy()
    img1 = cv2.resize(img1, (width, height))
    img1 = img1.astype(np.float32)

    ret, img = video.read()
    currentFrame += 1
    ori_img = img.copy()
    img = cv2.resize(img, (width, height))
    img = img.astype(np.float32)

    frame_num = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    tennis_loc_arr = np.full((frame_num, 2), None)

    while (True):
        ori_img2 = ori_img1
        img2 = img1
        ori_img1 = ori_img
        img1 = img
        video.set(1, currentFrame);
        ret, img = video.read()
        if not ret:
            break
        ori_img = img.copy()
        img = cv2.resize(img, (width, height))
        img = img.astype(np.float32)

        gray = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(ori_img1, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, gray2)
        ret, diff = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)

        X = np.concatenate((img, img1, img2), axis=2)
        X = np.rollaxis(X, 2, 0)
        input = torch.from_numpy(X).to(device)
        input = torch.unsqueeze(input, 0)
        with torch.no_grad():
            # pred = net(input)
            d1, d2, d3, d4, d5, d6, d7 = net(input)
            pred = d1[:, :, :, :]
        pred = pred.reshape((train_batch_size, input_height, input_width))
        heatmap = (pred * 255).cpu().numpy().astype(np.uint8)
        heatmap = np.array([cv2.resize(img, (output_width, output_height)) for img in heatmap])
        heatmap = heatmap * (diff/255)
        pred_circles, conf = get_heatmap_preds(torch.from_numpy(heatmap))
        pred_circles = pred_circles.numpy()[0]
        pred_has_circle = [np.max(img) > 0 for img in heatmap]
        if pred_has_circle[0]==True:
            tennis_loc_arr[currentFrame][0] = pred_circles[0]
            tennis_loc_arr[currentFrame][1] = pred_circles[1]
            print(f"Frame {currentFrame}: ({pred_circles[0]},{pred_circles[1]})")
        currentFrame += 1

    tennis_loc_arr = interpolation(tennis_loc_arr)
    tennis_loc_arr = np.array(tennis_loc_arr)
    vols = calculate_velocity(tennis_loc_arr)
    y_vol = vols[:, 1]
    x_vol = vols[:, 0]

    hit, start, end = jud_y_dir(y_vol, x_vol)

    seqlen = gt_hit.shape[0]
    # np.concatenate((hit[:, np.newaxis], gt_hit.astype(np.int64)[:, np.newaxis]), axis=1)
    for j in range(seqlen):
        if gt_hit[j] == 1:
            ALL_HAS += 1
            start = max(0, j - 12)
            end = min(seqlen, j + 12)
            if 1 in hit[start:end]:
                TP += 1
        if hit[j] == 1:
            start = max(0, j - 12)
            end = min(seqlen, j + 12)
            if 1 not in gt_hit[start:end]:
                FP += 1
    print("TP:", TP)
    print("FP:", FP)
    print("ALL_HAS:", ALL_HAS)
    print("Precision:", TP / (TP + FP))
    print("Recall:", TP / ALL_HAS)
    print('----------------------------------------------------------------')
