import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import os.path as osp
import argparse
import Models
import queue
import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from utils.utils import video_to_images, save_PIL_image, gen_tennis_loc_csv, save_np_image

# --save_weights_path=weights/model.0 --input_video_path="test.mp4" --output_video_path="test_TrackNet.mp4" --n_classes=256
# --save_weights_path=weights/model.0 --input_video_path="play.mp4" --output_video_path="play_TrackNet.mp4" --n_classes=256
# --save_weights_path=weights/model.0 --input_video_path="VideoInput/168000453563685.mp4" --n_classes=256
# --save_weights_path=weights/model.0 --input_video_path="output3.mp4" --output_video_path="output3_TrackNet.mp4" --n_classes=256
# --save_weights_path=weights/model.0 --input_video_path="tmp.mp4" --output_video_path="tmp_TrackNet.mp4" --n_classes=256

#parse parameters
parser = argparse.ArgumentParser()
parser.add_argument("--input_video_path", type=str)
parser.add_argument("--output_video_path", type=str, default="")
parser.add_argument("--save_weights_path", type=str)
parser.add_argument("--n_classes", type=int)

args = parser.parse_args()
input_video_path = args.input_video_path
output_video_path = args.output_video_path
save_weights_path = args.save_weights_path
n_classes = args.n_classes
device = 'cuda'
dst_folder = f'./tmp/{osp.basename(input_video_path)[:-4]}'

if output_video_path == "":
	#output video in same path
	output_video_path = input_video_path.split('.')[0].split('/')[1] + "_UIU.mp4"

#get video fps&video size
video = cv2.VideoCapture(input_video_path)
fps = int(video.get(cv2.CAP_PROP_FPS))
output_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
output_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

#start from first frame
currentFrame = 0

#width and height in TrackNet
width , height = 640, 360
img, img1, img2 = None, None, None

#load TrackNet model
# modelFN = Models.TrackNet.TrackNet
# m = modelFN( n_classes , input_height=height, input_width=width   )
# m.compile(loss='categorical_crossentropy', optimizer= 'adadelta' , metrics=['accuracy'])
# m.load_weights(  save_weights_path  )
from Models.uiunet import UIUNET
# from Models.TrackNet import TrackNet_pt
net = UIUNET(9, 1).to(device)
net.load_state_dict(torch.load('./weights/model.pt.uiu.best'), strict=True)

# In order to draw the trajectory of tennis, we need to save the coordinate of preious 7 frames
q = queue.deque()
for i in range(0,8):
	q.appendleft(None)

#save prediction images as vidoe
#Tutorial: https://stackoverflow.com/questions/33631489/error-during-saving-a-video-using-python-and-opencv
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter(dst_folder+f'/{output_video_path}',fourcc, fps, (output_width,output_height))

input_height=360
input_width=640
from LoadBatches import VideoDataset
from torch.utils.data import DataLoader
from utils.kp_utils import get_heatmap_preds
if not os.path.exists(dst_folder+'/imgs'):
	video_to_images(input_video_path, img_folder=dst_folder+'/imgs')
eval_dt = VideoDataset(video_path=input_video_path, n_classes=n_classes, input_height=360, input_width=640,
                          # output_height=input_height, output_width=input_width, num_images=64)
                          output_height=360, output_width=640, train=False, rt_ori=True)
data_loader = DataLoader(eval_dt, batch_size=2, shuffle=False, num_workers=8)
# data_loader = DataLoader(eval_dt, batch_size=2, shuffle=False, num_workers=0)

from tqdm import tqdm
pbar = tqdm(data_loader,
                # total=len(data_loader),
                total=len(data_loader))

frame_num = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
tennis_loc_arr = np.full((frame_num, 2), None)
currentFrame=2


net.eval()
for step, batch in enumerate(pbar):
	if step<2:
		continue
	input, img_ori = batch
	rt_ori = img_ori.numpy()
	ori_img = rt_ori[:, :, :, 0:3]
	ori_img1 = rt_ori[:, :, :, 3:6]
	batch_size = input.shape[0]
	gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in ori_img]
	gray2 = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in ori_img1]
	diff = [cv2.absdiff(img1, img2) for img1, img2 in zip(gray, gray2)]
	diff = [cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)[1] for img in diff]
	diff = np.array(diff)
	with torch.no_grad():
		# pr = (net(batch.to('cuda')))
		d1, d2, d3, d4, d5, d6, d7 = (net(input.to('cuda')))
		pred = d1[:, :, :, :]

	pred = pred.reshape((batch_size, input_height, input_width))
	heatmap = (pred * 255).cpu().numpy().astype(np.uint8)
	heatmap = np.array([cv2.resize(img, (output_width, output_height)) for img in heatmap])
	heatmap = heatmap * (diff / 255)
	pred_circles, conf = get_heatmap_preds(torch.from_numpy(heatmap))
	pred_circles = pred_circles.numpy()
	pred_has_circle = [np.max(img) > 0 for img in heatmap]
	# pred_has_circle = [np.max(img) > 50 for img in heatmap]
	for j, pred_has in enumerate(pred_has_circle):
		output_img = ori_img[j]
		save_np_image(img=heatmap, img_folder=dst_folder + '/htm', img_name="{:06d}.png".format(currentFrame))
		PIL_image = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
		PIL_image = Image.fromarray(PIL_image)
		if pred_has==True:
			draw_x = pred_circles[j][0]
			draw_y = pred_circles[j][1]
			tennis_loc_arr[currentFrame][0] = draw_x
			tennis_loc_arr[currentFrame][1] = draw_y
			bbox = (draw_x - 6, draw_y - 6, draw_x + 6, draw_y + 6)
			draw = ImageDraw.Draw(PIL_image)
			draw.rectangle(bbox, outline='red', width=2)
			del draw
		opencvImage = cv2.cvtColor(np.array(PIL_image), cv2.COLOR_RGB2BGR)
		save_np_image(img=opencvImage, img_folder=dst_folder + '/bbox', img_name="{:06d}.png".format(currentFrame))
		output_video.write(opencvImage)
		currentFrame += 1

gen_tennis_loc_csv(dst_folder, tennis_loc_arr)

output_video.release()
print("finish")
