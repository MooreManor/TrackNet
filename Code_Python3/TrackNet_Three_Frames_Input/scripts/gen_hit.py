import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import pandas as pd
from sktime.datatypes._panel._convert import from_2d_array_to_nested
import subprocess
import glob
from utils.utils import calculate_velocity, add_csv_col, jud_dir, add_text_to_video
import csv
import numpy as np
from pickle import load

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
  x, y = [x[0] if x[0] is not None else np.nan for x in coords], [x[1] if x[1] is not None else np.nan for x in coords]

  xxx = np.array(x) # x coords
  yyy = np.array(y) # y coords

  nons, yy = nan_helper(xxx)
  xxx[nons]= np.interp(yy(nons), yy(~nons), xxx[~nons])
  nans, xx = nan_helper(yyy)
  yyy[nans]= np.interp(xx(nans), xx(~nans), yyy[~nans])

  newCoords = [*zip(xxx,yyy)]

  return newCoords


# vids = glob.glob('VideoInput/*.mp4')
# vids = ['dataset4.mp4']
vids = ['1689861601046.mp4']
from classfiers.resnet import Classifier_RESNET_pt
net = Classifier_RESNET_pt(2, 8).to('cuda')

for vid in vids:
    dir = vid.split('/')[-1][:-4]
    dir_path = './tmp/'+dir
    # mp4_path = os.path.join(dir_path, dir+'_TrackNet.mp4')
    mp4_path = os.path.join(dir_path, dir+'_UIU.mp4')
    output_path = os.path.join(dir_path, dir+'_hit.mp4')
    with open(dir_path+'/tennis.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader) # 跳过第一行
        data = []
        for row in reader:
            if row[1]=='None':
                x, y = None, None
            else:
                x, y = float(row[1]), float(row[2])
            data.append([x, y])
    data = interpolation(data)
    data = np.array(data)
    vols = calculate_velocity(data)
    y_vol = vols[:, 1]
    x_vol = vols[:, 0]
    V=[]
    # for i in range(len(vols)-1):
    for i in range(len(vols)):
        vx = vols[i, 0]
        vy = vols[i, 1]
        if vx==None:
            v=0
        else:
            v = (vx ** 2 + vy ** 2) ** 0.5
        V.append(v)

    hit, start, end = jud_dir(y_vol, x_vol)
    add_text_to_video(vis=0, input_video_path=mp4_path,
                      output_video_path=output_path, hit=hit, start=start, end=end, speed=V, xy=data)
    # command = ['python',
    #                'predict_video_bbox.py',
    #                 '--save_weights_path=weights/model.3',
    #                 f'--input_video_path={vid}',
    #                '--n_classes=256']
    # print(f'Running \"{" ".join(command)}\"')
    # subprocess.call(command)


# 读取csv文件



