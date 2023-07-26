import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from utils.utils import calculate_velocity, add_csv_col, jud_dir, add_text_to_video
import csv
import numpy as np
from utils.vis_utils import plot_wave_scat

# 读取csv文件
with open('./gt.csv', newline='', encoding='gbk') as csvfile:
    reader = csv.reader(csvfile)
    next(reader) # 跳过第一行
    data = []
    for row in reader:
        if len(row)==1:
            continue
        else:
            x, y = float(row[1]), float(row[2])
            bounce = int(row[4])
            hit = int(row[3])
        data.append([x, y, bounce, hit])

# 将数据转换为numpy数组
data = np.array(data)
velocities = []
velocities.append((0, 0))
for i in range(1, data.shape[0]):
    velocities.append((data[i][0]-data[i-1][0], data[i][1]-data[i-1][1]))

velocities = np.array(velocities)
sum_velo = pow(pow(velocities[:, 0], 2)+pow(velocities[:, 1], 2), 0.5)

acceleration = []
acceleration.append((0, 0))
for i in range(1, len(velocities)):
    acc = (velocities[i] - velocities[i-1])
    acceleration.append(acc)
acceleration = np.array(acceleration)
sum_accel = pow(pow(acceleration[:, 0], 2)+pow(acceleration[:, 1], 2), 0.5)
bounce = data[:, 2].astype(np.int)
hit = data[:, 3].astype(np.int)
plot_wave_scat(velocities[:, 0], hit, ylabel='x vol', title='relation between vol and hit')
plot_wave_scat(velocities[:, 1], hit, ylabel='y vol', title='relation between vol and hit')
plot_wave_scat(data[:, 0], hit, ylabel='x pos', title='relation between pos and hit')
plot_wave_scat(data[:, 1], hit, ylabel='y pos', title='relation between pos and hit')
plot_wave_scat(acceleration[:, 0], hit, ylabel='x accel', title='relation between accel and hit')
plot_wave_scat(acceleration[:, 1], hit, ylabel='y accel', title='relation between accel and hit')
plot_wave_scat(sum_velo, hit, ylabel='vol', title='relation between vol and hit')
plot_wave_scat(sum_accel, hit, ylabel='accel', title='relation between accel and hit')