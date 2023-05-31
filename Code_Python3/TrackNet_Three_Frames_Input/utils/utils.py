import os
import os.path as osp
import subprocess
import cv2

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

def gen_tennis_loc_csv(folder, data=None, file_name='tennis_loc.csv'):
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

            writer.writerow([frame, x, y])


