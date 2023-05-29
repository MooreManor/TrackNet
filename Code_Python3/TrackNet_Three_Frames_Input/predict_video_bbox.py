import argparse
import Models
import queue
import cv2
import numpy as np
from PIL import Image, ImageDraw
from utils import video_to_images, save_PIL_image, gen_tennis_loc_csv
import os.path as osp


# --save_weights_path=weights/model.3 --input_video_path="test.mp4" --output_video_path="test_TrackNet.mp4" --n_classes=256
# --save_weights_path=weights/model.3 --input_video_path="play.mp4" --output_video_path="play_TrackNet.mp4" --n_classes=256
# --save_weights_path=weights/model.3 --input_video_path="VideoInput/167979954199057.mp4" --output_video_path="tmp_TrackNet.mp4" --n_classes=256
# --save_weights_path=weights/model.3 --input_video_path="output3.mp4" --output_video_path="output3_TrackNet.mp4" --n_classes=256
# --save_weights_path=weights/model.3 --input_video_path="tmp.mp4" --output_video_path="tmp_TrackNet.mp4" --n_classes=256
# --save_weights_path=weights/model.0.1 --input_video_path="output3.mp4" --output_video_path="output3_TrackNet.mp4" --n_classes=256

#parse parameters
parser = argparse.ArgumentParser()
parser.add_argument("--input_video_path", type=str)
parser.add_argument("--output_video_path", type=str, default = "")
parser.add_argument("--save_weights_path", type = str  )
parser.add_argument("--n_classes", type=int )

args = parser.parse_args()
input_video_path =  args.input_video_path
output_video_path =  args.output_video_path
save_weights_path = args.save_weights_path
n_classes =  args.n_classes

if output_video_path == "":
	#output video in same path
	output_video_path = input_video_path.split('.')[0] + "_TrackNet.mp4"

#get video fps&video size
print('Turning the videos to images...')
dst_folder = f'./tmp/{osp.basename(input_video_path)[:-4]}'
video_to_images(input_video_path, img_folder=dst_folder+'/imgs')
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
modelFN = Models.TrackNet.TrackNet
m = modelFN( n_classes , input_height=height, input_width=width   )
m.compile(loss='categorical_crossentropy', optimizer= 'adadelta' , metrics=['accuracy'])
m.load_weights(  save_weights_path  )

# In order to draw the trajectory of tennis, we need to save the coordinate of preious 7 frames
q = queue.deque()
for i in range(0,8):
	q.appendleft(None)

#save prediction images as vidoe
#Tutorial: https://stackoverflow.com/questions/33631489/error-during-saving-a-video-using-python-and-opencv
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video_path = osp.join(dst_folder, output_video_path)
output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width,output_height))


#both first and second frames cant be predict, so we directly write the frames to output video
#capture frame-by-frame
video.set(1,currentFrame);
ret, img1 = video.read()
#write image to video
output_video.write(img1)
currentFrame +=1
#resize it
ori_img1 = img1.copy()
img1 = cv2.resize(img1, ( width , height ))
#input must be float type
img1 = img1.astype(np.float32)

#capture frame-by-frame
video.set(1,currentFrame);
ret, img = video.read()
#write image to video
output_video.write(img)
currentFrame +=1
#resize it
ori_img = img.copy()
img = cv2.resize(img, ( width , height))
#input must be float type
img = img.astype(np.float32)

gen_tennis_loc_csv(dst_folder)

x=y=0
while(True):

	ori_img2 = ori_img1
	img2 = img1
	ori_img1 = ori_img
	img1 = img

	#capture frame-by-frame
	video.set(1,currentFrame);
	ret, img = video.read()

	#if there dont have any frame in video, break
	if not ret:
		break
	ori_img = img.copy()

	#img is the frame that TrackNet will predict the position
	#since we need to change the size and type of img, copy it to output_img
	output_img = img

	#resize it
	img = cv2.resize(img, ( width , height ))
	#input must be float type
	img = img.astype(np.float32)


	#combine three imgs to  (width , height, rgb*3)
	X = np.concatenate((img, img1, img2),axis=2)

	#since the odering of TrackNet  is 'channels_first', so we need to change the axis
	X = np.rollaxis(X, 2, 0)
	#prdict heatmap
	# import time
	# start = time.time()
	pr = m.predict( np.array([X]) )[0]
	# print('Last time:', time.time()-start)

	#since TrackNet output is ( net_output_height*model_output_width , n_classes )
	#so we need to reshape image as ( net_output_height, model_output_width , n_classes(depth) )
	#.argmax( axis=2 ) => select the largest probability as class
	pr = pr.reshape(( height ,  width , n_classes ) ).argmax( axis=2)

	#cv2 image must be numpy.uint8, convert numpy.int64 to numpy.uint8
	pr = pr.astype(np.uint8)

	# #reshape the image size as original input image
	# 将两张图像转换为灰度图像
	gray = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
	gray2 = cv2.cvtColor(ori_img1, cv2.COLOR_BGR2GRAY)

	# 计算两张图像的差异
	diff = cv2.absdiff(gray, gray2)
	heatmap = cv2.resize(pr, (output_width, output_height))
	ret, diff = cv2.threshold(diff,5,255,cv2.THRESH_BINARY)
	# heatmap = heatmap * diff
	ori_heatmap = heatmap.copy()

	#heatmap is converted into a binary image by threshold method.
	ret,heatmap = cv2.threshold(heatmap,127,255,cv2.THRESH_BINARY)

	#find the circle in image with 2<=radius<=7
	circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT,dp=1,minDist=1,param1=50,param2=2,minRadius=2,maxRadius=7)
	# if circles is None and x>0 and y>0:
	# 	# 获取图像中所有值大于0的像素点的坐标
	# 	coords = np.column_stack(np.where(heatmap > 0))
	# 	if coords
	#
	# 	# 根据目标点，计算所有像素点到目标点的距离
	# 	distances = np.sqrt((coords[:, 0] - y) ** 2 + (coords[:, 1] - x) ** 2)
	#
	# 	# 找到距离最小的像素点的索引
	# 	min_index = np.argmin(distances)
	#
	# 	# 获取距离最小的像素点的坐标，即连通域的中心点
	# 	center_point = coords[min_index]
	#
	# 	circles = center_point

	#In order to draw the circle in output_img, we need to used PIL library
	#Convert opencv image format to PIL image format
	# delta = ori_img - ori_img1
	# from matplotlib import pyplot as plt
	# plt.imshow(delta)
	# plt.show()

	# ssim_score, diff = cv2.compareStructures(gray, gray2, win_size=(11, 11), sigma=1.5, K1=0.01, K2=0.03, L=255)


	# # 对差异图像进行二值化处理
	# thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)[1]
	#
	# # 使用形态学操作去除噪声
	# kernel = np.ones((5, 5), np.uint8)
	# thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

	# from matplotlib import pyplot as plt
	# plt.imshow(diff)
	# plt.show()

	# window_size = 10
	# half_window = window_size // 2
	# if circles is not None:
	# 	for i in range(circles.shape[1]):
	# 		circle = circles[0][i]
	# 		x = circle[0]
	# 		y = circle[1]
	# 		window = diff[int(y - half_window):int(y + half_window + 1), int(x - half_window):int(x + half_window + 1)]
	# 		# if np.sum(window)<100:
	# 		if np.sum(window)<300:
	# 			circles[0][i][2] = -1


	# if circles is not None:
	# 	for i in range(circles.shape[1]):
	# 		circle = circles[0][i]
	# 		bbox = (circle[0] - 6, circle[1] - 6, circle[0] + 6, circle[1] + 6)
	# 		cv2.rectangle(output_img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (18, 127, 15), thickness=2)
	PIL_image = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
	PIL_image = Image.fromarray(PIL_image)

	#check if there have any tennis be detected
	# from matplotlib import pyplot as plt
	# plt.imshow(PIL_image)
	# plt.show()
	if circles is not None:
		#if only one tennis be detected
		if len(circles) == 1:
			for i in range(circles.shape[1]):
				if(circles[0][i][2] < 0):
					continue
				x = int(circles[0][i][0])
				y = int(circles[0][i][1])
				break

			# x = int(circles[0][0][0])
			# y = int(circles[0][0][1])
			print(currentFrame, x,y)

			#push x,y to queue
			q.appendleft([x,y])
			#pop x,y from queue
			q.pop()
		else:
			#push None to queue
			q.appendleft(None)
			#pop x,y from queue
			q.pop()
	else:
		#push None to queue
		q.appendleft(None)
		#pop x,y from queue
		q.pop()

	#draw current frame prediction and previous 7 frames as yellow circle, total: 8 frames
	# flag = 0
	for i in range(0,1):
		if q[i] is not None:
			draw_x = q[i][0]
			draw_y = q[i][1]
			bbox = (draw_x - 6, draw_y - 6, draw_x + 6, draw_y + 6)
			draw = ImageDraw.Draw(PIL_image)
			# draw.ellipse(bbox, outline ='yellow')
			draw.rectangle(bbox, outline='red', width=2)
			save_PIL_image(PIL_img=PIL_image, img_folder=dst_folder+'/bbox', img_name="{:06d}.png".format(currentFrame))
			del draw
			# flag = 1

	#Convert PIL image format back to opencv image format
	opencvImage = cv2.cvtColor(np.array(PIL_image), cv2.COLOR_RGB2BGR)
	#write image to output_video
	output_video.write(opencvImage)

	#next frame
	currentFrame += 1

# everything is done, release the video
video.release()
output_video.release()
print("finish")