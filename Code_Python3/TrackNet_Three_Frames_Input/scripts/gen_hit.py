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
vids = ['168114343892521.mp4']

for vid in vids:
    dir = vid.split('/')[-1][:-4]
    dir_path = './tmp/'+dir
    mp4_path = os.path.join(dir_path, dir+'_TrackNet.mp4')
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
    # test_df = pd.DataFrame({'x': [coord[0] for coord in data[:-1]], 'y': [coord[1] for coord in data[:-1]], 'V': V})
    #
    # # df.shift
    # for i in range(20, 0, -1):
    #     test_df[f'lagX_{i}'] = test_df['x'].shift(i, fill_value=0)
    # for i in range(20, 0, -1):
    #     test_df[f'lagY_{i}'] = test_df['y'].shift(i, fill_value=0)
    # for i in range(20, 0, -1):
    #     test_df[f'lagV_{i}'] = test_df['V'].shift(i, fill_value=0)
    #
    # test_df.drop(['x', 'y', 'V'], 1, inplace=True)
    #
    # Xs = test_df[['lagX_20', 'lagX_19', 'lagX_18', 'lagX_17', 'lagX_16',
    #               'lagX_15', 'lagX_14', 'lagX_13', 'lagX_12', 'lagX_11', 'lagX_10',
    #               'lagX_9', 'lagX_8', 'lagX_7', 'lagX_6', 'lagX_5', 'lagX_4', 'lagX_3',
    #               'lagX_2', 'lagX_1']]
    # Xs = from_2d_array_to_nested(Xs.to_numpy())
    #
    # Ys = test_df[['lagY_20', 'lagY_19', 'lagY_18', 'lagY_17',
    #               'lagY_16', 'lagY_15', 'lagY_14', 'lagY_13', 'lagY_12', 'lagY_11',
    #               'lagY_10', 'lagY_9', 'lagY_8', 'lagY_7', 'lagY_6', 'lagY_5', 'lagY_4',
    #               'lagY_3', 'lagY_2', 'lagY_1']]
    # Ys = from_2d_array_to_nested(Ys.to_numpy())
    #
    # Vs = test_df[['lagV_20', 'lagV_19', 'lagV_18',
    #               'lagV_17', 'lagV_16', 'lagV_15', 'lagV_14', 'lagV_13', 'lagV_12',
    #               'lagV_11', 'lagV_10', 'lagV_9', 'lagV_8', 'lagV_7', 'lagV_6', 'lagV_5',
    #               'lagV_4', 'lagV_3', 'lagV_2', 'lagV_1']]
    # Vs = from_2d_array_to_nested(Vs.to_numpy())
    #
    # X = pd.concat([Xs, Ys, Vs], 1)
    #
    # # load the pre-trained classifier
    # clf = load(open('clf.pkl', 'rb'))
    #
    # predcted = clf.predict(X)
    # idx = list(np.where(predcted == 1)[0])
    # idx = np.array(idx) - 10
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



