import os
import cv2
import random

# 设置参数
video_dir = '/home/liziqing/Downloads/哈啰'  # 视频文件夹路径
sample_num = 20000  # 抽取的总帧数
exclude_vids = ['26048.mp4', '28158.mp4']
sample_per_video = sample_num // len([f for f in os.listdir(video_dir) if f.endswith('.mp4') and f not in exclude_vids])  # 每个视频抽取的帧数
output_vid_dir = '/datasetb/tennis/sample/vids'  # 输出文件夹路径
output_img_dir = '/datasetb/tennis/sample/imgs'  # 输出文件夹路径
subdirectory_size = 1000

# 遍历视频文件夹
for video_file in os.listdir(video_dir):
    if not video_file.endswith('.mp4') or video_file in exclude_vids:
        continue

    # 打开视频文件
    video_path = os.path.join(video_dir, video_file)
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 确定抽取的帧数和位置
    # sample_indices = random.sample(range(frame_count), sample_per_video)
    start_idx = random.randint(0, frame_count - sample_per_video)
    sample_indices = range(start_idx, start_idx + sample_per_video)

    # 遍历每个需要抽取的帧
    for i, frame_idx in enumerate(sample_indices):
        # 读取帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        # 保存视频帧
        output_video_path = os.path.join(output_vid_dir, video_file.replace('.mp4', '_sample.mp4'))
        if i == 0:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
            out_video = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)
        out_video.write(frame)

        # 保存图片
        # output_image_path = os.path.join(output_img_dir, video_file.replace('.mp4', ''), f'{frame_idx}.png')
        output_image_path = os.path.join(output_img_dir, video_file.replace('.mp4', ''), "{:03d}".format(frame_idx//subdirectory_size), "{:06d}.png".format(frame_idx))
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        cv2.imwrite(output_image_path, frame)

    # 释放视频文件
    cap.release()
    out_video.release()