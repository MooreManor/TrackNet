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

def save_np_image(img, img_name, img_folder):
    os.makedirs(img_folder, exist_ok=True)
    cv2.imwrite(osp.join(img_folder, img_name), img)

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
                # if positions[j][0] != None:
                if positions[j][0] != None and positions[i][0] - positions[j][0]<50 and positions[i][1] - positions[j][1]<50:
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

def find_start_of_hit(lst, value):
    result = []
    start = None
    for i, x in enumerate(lst):
        if x == value:
            if start is None:
                start = i
                result.append(start)
            # elif lst[i-1] != 1:
            #     result.append(start)
            #     start = i
        elif x == -value:
            start = None
    # if start is not None:
    #     result.append(start)
    return result

def jud_dir(y_vols, x_vols,interval=4):
    frame_num = y_vols.shape[0]
    # i = 4
    # hit = np.full((frame_num, 3), None)
    hit = np.zeros(frame_num, dtype=int)
    final = np.zeros(frame_num, dtype=int)

    for i in range(4, frame_num - 3):
        window = y_vols[i:i + 4]
        window_x = x_vols[i:i + 4]
        signs = np.array([np.sign(win) for win in window if win is not None])
        # abs_values = np.array([np.abs(win) for win in window if win is not None])
        # big = np.all(abs_values > 10) and len(abs_values)>0
        values = np.array([win for win in window if win is not None])
        x_values = np.array([win for win in window_x if win is not None])
        big1 = np.all(values > 3) and len(values)>0 # 对面的
        big2 = np.all(values < -10) and len(values)>0
        x_abs_values = np.array([np.abs(win) for win in x_values if win is not None])
        x_big = np.all(x_abs_values > 1) and len(x_abs_values)>0
        big = big1+big2
        if np.all(signs > 0):
            hit[i] = 1  # 正符号
        elif np.all(signs < 0):
            hit[i] = -1  # 负符号
        hit[i] *= big
        hit[i] *= x_big

    index_1 = np.where(hit == 1)[0][0]
    index_2 = np.where(hit == -1)[0][0]
    # index_3 = [i for i, x in enumerate(hit) if x == 1 and (i == 0 or hit[i-1] != 1)][-1]
    index_3 = find_start_of_hit(hit, 1)[-1]
    index_4 = find_start_of_hit(hit, -1)[-1]
    end = index_3 if index_3 > index_4 else index_4
    if index_1 < index_2:
        start = index_1
        dir =1
    else:
        start = index_2
        dir = -1
    final[start] = 1
    for i in range(start, frame_num):
        if hit[i] == -dir:
            final[i] = 1
            dir *= -1

    return final, start, end
    # while i < frame_num-interval:
    #
    #     i += 1

def add_text_to_frame(frame, text, position= (10, 100), color=(0, 0, 255), thickness=8, font_scale=4):
    # 在左上角添加文本
    font = cv2.FONT_HERSHEY_SIMPLEX
    frame_with_text = cv2.putText(frame, text, position, font, font_scale, color, thickness)
    return frame_with_text

# def add_text_to_video(input_video_path, output_video_path, hit, start, start_frame, end_frame):
def add_text_to_video(input_video_path, output_video_path, hit, start, end, speed, xy,vis=0):
    video = cv2.VideoCapture(input_video_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    output_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    output_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height))
    currentFrame = 0
    while(True):
        ret, img = video.read()
        # if there dont have any frame in video, break
        if not ret:
            break
        img = add_text_to_frame(img, "Speed: {:.2f}".format(speed[currentFrame//10*10]), position=(1000, 25), color=(0, 255, 255), thickness=2, font_scale=1)
        if start <= currentFrame <= start+10:
            img = add_text_to_frame(img, "Start", position=(800, 100), color=(0, 255, 0), font_scale=2)
        if end <= currentFrame <= end+10:
            img = add_text_to_frame(img, "End", position=(800, 100), color=(0, 255, 0), font_scale=2)
        tmp = max(0, currentFrame-6)
        if 1 in hit[tmp: currentFrame] :
            hit_frame = tmp+np.where(hit[tmp: currentFrame]==1)[0][0]
            img = add_text_to_frame(img, f"Hit:({int(xy[hit_frame][0])},{int(xy[hit_frame][1])})", position=(800, 150), font_scale=1.5)
        if vis:
            save_np_image(img=img, img_folder=os.path.dirname(output_video_path) + '/hit', img_name="{:06d}.png".format(currentFrame))
        # img = add_text_to_frame(img, f"Speed: {speed[currentFrame//10*10]}", position=(200, 100), color=(0, 0, 255), thickness=4)
        output_video.write(img)

        currentFrame += 1

    video.release()
    output_video.release()
