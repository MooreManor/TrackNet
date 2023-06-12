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
    fieldnames = ['帧数', 'x坐标', 'y坐标', '置信度']

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
            s = data[i][2]
            if x is None:
                x = 'None'
            if y is None:
                y = 'None'
            if s is None:
                s = 'None'

            writer.writerow([frame, x, y, s])

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

def vis_flow(flow, gray1):
    # 可视化光流场
    flow_x = flow[..., 0]
    flow_y = flow[..., 1]

    # 创建画布
    h, w = gray1.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # 画箭头表示光流向量
    step = 10  # 控制箭头密度
    for y in range(0, h, step):
        for x in range(0, w, step):
            dx = int(flow_x[y, x])
            dy = int(flow_y[y, x])
            cv2.arrowedLine(canvas, (x, y), (x + dx, y + dy), (0, 255, 0), 1, tipLength=0.5)
    cv2.imshow('Optical Flow', canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 光流限制范围
FLOW_REG = 100
FLOW_THR = 2
CONF_THR = 0.3
def judge_in_square(q_point_x, q_point_y, cen_x, cen_y, side):
    if q_point_x >= cen_x-int(side/2) and q_point_x < cen_x+int(side/2)+1 and q_point_y >= cen_y-int(side/2) and q_point_y < cen_y+int(side/2)+1:
        return True
    return False
def pos_pred_flow(gray, gray2, update_step, current_frame, output_width, output_height, steady_x=None, steady_y=None, predict_x=None, predict_y=None):
    flow = cv2.calcOpticalFlowFarneback(gray, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # vis_flow(flow, gray)
    flow_val = np.linalg.norm(flow, axis=-1)
    change_steady = False
    # 有光流且在搜索范围内
    if predict_x!=None and flow_val[predict_y][predict_x]>FLOW_THR:
        # if update_step is not None and judge_in_square(predict_x, predict_y, steady_x, steady_y, FLOW_REG):
        if update_step is not None:
            change_steady = True
            return predict_y, predict_x, change_steady

    # 寻找光流中心
    if update_step is not None:
        flow_x = steady_x
        flow_y = steady_y
    else:
        flow_x = predict_x
        flow_y = predict_y
    # 计算小区域的左上角和右下角坐标
    left = max(flow_x - int(FLOW_REG / 2), 0)
    top = max(flow_y - int(FLOW_REG / 2), 0)
    right = min(flow_x + int(FLOW_REG / 2) + 1, output_width)
    bottom = min(flow_y + int(FLOW_REG / 2) + 1, output_height)
    flow_array = flow_val[top:bottom, left:right]
    # ret, heatmap = cv2.threshold(flow_array, 1, 255, cv2.THRESH_BINARY)
    # heatmap = heatmap.astype(np.uint8)
    # circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=10, param1=50, param2=2, minRadius=2,
    #                            maxRadius=7)

    max_pos = np.argmax(flow_array)
    flow_max_y, flow_max_x = np.unravel_index(max_pos, flow_array.shape)
    flow_max_y = flow_max_y + top
    flow_max_x = flow_max_x + left
    return flow_max_y, flow_max_x, change_steady

def Get_HSV(image):
    # 1 get trackbar's value
    hmin = cv2.getTrackbarPos('hmin', 'h_binary')
    hmax = cv2.getTrackbarPos('hmax', 'h_binary')
    smin = cv2.getTrackbarPos('smin', 's_binary')
    smax = cv2.getTrackbarPos('smax', 's_binary')
    vmin = cv2.getTrackbarPos('vmin', 'v_binary')
    vmax = cv2.getTrackbarPos('vmax', 'v_binary')

    # 2 to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (3, 3), 0)  # denoise
    cv2.imshow('hsv', hsv)
    h, s, v = cv2.split(hsv)

    # 3 set threshold (binary image)
    # if value in (min, max):white; otherwise:black
    h_binary = cv2.inRange(np.array(h), np.array(hmin), np.array(hmax))
    s_binary = cv2.inRange(np.array(s), np.array(smin), np.array(smax))
    v_binary = cv2.inRange(np.array(v), np.array(vmin), np.array(vmax))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # ellipse kernel
    h_binary = cv2.morphologyEx(h_binary, cv2.MORPH_CLOSE, kernel)
    s_binary = cv2.morphologyEx(s_binary, cv2.MORPH_CLOSE, kernel)
    v_binary = cv2.morphologyEx(v_binary, cv2.MORPH_CLOSE, kernel)

    # 4 get binary（对H、S、V三个通道分别与操作）
    binary = cv2.bitwise_and(h_binary, cv2.bitwise_and(s_binary, v_binary))

    # 6 Show
    cv2.imshow('h_binary', h_binary)
    cv2.imshow('s_binary', s_binary)
    cv2.imshow('v_binary', v_binary)
    cv2.imshow('binary', binary)

    return binary

def hsv_thr_vis(img_name):
    # create trackbars
    cv2.namedWindow('h_binary')
    cv2.createTrackbar('hmin', 'h_binary', 0, 255, lambda x: None)
    cv2.createTrackbar('hmax', 'h_binary', 255, 255, lambda x: None)

    cv2.namedWindow('s_binary')
    cv2.createTrackbar('smin', 's_binary', 0, 255, lambda x: None)
    cv2.createTrackbar('smax', 's_binary', 255, 255, lambda x: None)

    cv2.namedWindow('v_binary')
    cv2.createTrackbar('vmin', 'v_binary', 0, 255, lambda x: None)
    cv2.createTrackbar('vmax', 'v_binary', 255, 255, lambda x: None)

    # load image
    image = cv2.imread(img_name)

    while True:
        binary = Get_HSV(image)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # press 'ESC' to exit
            break

    cv2.destroyAllWindows()
