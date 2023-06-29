import os
import os.path as osp
import subprocess
import cv2
import numpy as np

def video_to_images(vid_file, img_folder=None, return_info=False):
    if img_folder is None:
        img_folder = osp.join('/tmp', osp.basename(vid_file).replace('.', '_'))

    os.makedirs(img_folder, exist_ok=True)

    command = ['ffmpeg',
               '-i', vid_file,
               '-f', 'image2',
               '-v', 'error',
               '-start_number', '0',
               f'{img_folder}/%06d.png']
    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)

    print(f'Images saved to \"{img_folder}\"')

    img_shape = cv2.imread(osp.join(img_folder, '000001.png')).shape

    if return_info:
        return img_folder, len(os.listdir(img_folder)), img_shape
    else:
        return img_folder

def save_PIL_image(PIL_img, img_name, img_folder):
    os.makedirs(img_folder, exist_ok=True)
    PIL_img.save(osp.join(img_folder, img_name))

def gen_tennis_loc_csv(folder, data=None, file_name='tennis.csv'):
    import csv
    # 定义 CSV 文件路径和字段名
    csv_file_path = osp.join(folder, file_name)
    fieldnames = ['帧数', 'x坐标', 'y坐标']

    # 创建 CSV 文件并写入字段名
    with open(csv_file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # writer.writeheader()
        writer.writerow(fieldnames)
        # 逐行写入数据
        for i in range(data.shape[0]):
            frame = i
            x = data[i][0]
            y = data[i][1]
            if x is None:
                x = 'None'
            if y is None:
                y = 'None'

            writer.writerow([frame, x, y])

def read_tennis_loc_csv(folder, file_name='tennis.csv'):
    import csv
    # 定义 CSV 文件路径和字段名
    csv_file_path = osp.join(folder, file_name)
    with open(csv_file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # 跳过第一行
        data = []
        for row in reader:
            x, y = float(row[1]), float(row[2])
            data.append([x, y])

    # 将数据转换为numpy数组
    data = np.array(data)
    return data

def gen_court_inf(folder, data):
    # 打开文件并写入多个线的坐标
    txt_file_path = osp.join(folder, 'lines.txt')
    with open(txt_file_path, 'w') as f:
        for i in range(0, len(data), 4):
            coords = f'{data[i]},{data[i + 1]},{data[i + 2]},{data[i + 3]}'
            f.write(coords + '\n')
def read_court_inf(folder):
    # 读取 gen_court_inf 函数生成的 txt 文件，返回线的坐标列表
    txt_file_path = osp.join(folder, 'lines.txt')
    lines = []
    with open(txt_file_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line:
                coords = line.split(',')
                lines.append([float(coord) for coord in coords])
    return lines


def add_csv_col(folder, new_col_name, data=None, file_name='tennis_loc.csv'):
    import csv
    csv_file_path = osp.join(folder, file_name)

    # 读取csv文件
    with open(csv_file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)

    # 添加第四列数据
    rows[0].append(new_col_name)
    for i in range(1, len(rows)):
        # 在这里填写添加第四列数据的代码
        # 假设新的数据保存在变量new_data中
        new_data = data[i-1]
        if new_data is None:
            new_data = 'None'
        rows[i].append(new_data)

    # 将更新后的数据写回到csv文件
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows)

def calculate_velocity(positions):
    velocities = []
    # accelerations = []
    velocities.append((None, None))
    # accelerations.append(-1, -1)
    # accelerations.append(-1, -1)
    for i in range(1, len(positions)):
        # 计算相邻两帧之间的距离和时间间隔
        if positions[i][0]==None:
            velocities.append((None, None))
        else:
            flag = 0
            for j in range(i - 1, -1, -1):
                if positions[j][0] != None:
                    dx = positions[i][0] - positions[j][0]
                    dy = positions[i][1] - positions[j][1]
                    flag=1
                    break
            if flag==1:
                dt = i-j
                # 计算相邻两帧之间的2D速度
                v = (dx / dt, dy / dt)
                velocities.append(v)
            else:
                velocities.append((None, None))

        # dx = positions[i][0] - positions[i - 1][0]
        # dy = positions[i][1] - positions[i - 1][1]
        # dt = 1  # 假设每帧的时间间隔为1
        #
        # # 计算相邻两帧之间的2D速度
        # v = (dx / dt, dy / dt)
        # velocities.append(v)

        # if i > 1:
        #     # 计算相邻两帧之间的2D加速度
        #     dvx = v[0] - velocities[i - 2][0]
        #     dvy = v[1] - velocities[i - 2][1]
        #     a = (dvx / dt, dvy / dt)
        #     accelerations.append(a)

    # return velocities, accelerations
    return np.array(velocities)
