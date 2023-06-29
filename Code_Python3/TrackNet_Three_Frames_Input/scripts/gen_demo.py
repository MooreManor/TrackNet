import subprocess
import glob

vids = glob.glob('VideoInput/*.mp4')
# --save_weights_path=weights/model.3 --input_video_path="output3.mp4" --n_classes=256
for vid in vids:
    command = ['python',
                   'predict_video_bbox.py',
                    '--save_weights_path=weights/model.3',
                    f'--input_video_path={vid}',
                   '--n_classes=256']
    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)