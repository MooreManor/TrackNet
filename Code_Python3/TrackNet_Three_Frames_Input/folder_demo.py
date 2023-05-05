import argparse
import Models
import queue
import cv2
import numpy as np
from PIL import Image, ImageDraw
import time
import glob
import os
import subprocess

# --save_weights_path=weights/model.3 --input_video_path="test.mp4" --output_video_path="test_TrackNet.mp4" --n_classes=256
# --save_weights_path=weights/model.3 --input_video_path="play.mp4" --output_video_path="play_TrackNet.mp4" --n_classes=256
video_input_folder = 'VideoInput'
video_output_folder = 'VideoOutput'
video_files = glob.glob(os.path.join(video_input_folder, "*.mp4"))

for video_file in video_files:
    # print(video_file)
    output_file = video_output_folder + '/' + os.path.basename(video_file[:-4]) +'_TrackNet.mp4'
    command = ['python', 'predict_video.py',
               '--save_weights_path=weights/model.3',
               f"--input_video_path={video_file}",
               f'--output_video_path={output_file}',
               '--n_classes=256']
    print(f'Running \"{" ".join(command)}\"')
    start = time.time()
    subprocess.call(command)
    print(f'{video_file} Cost Time {time.time()-start}')
