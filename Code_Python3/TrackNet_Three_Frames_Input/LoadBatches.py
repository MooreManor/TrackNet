import numpy as np
import cv2
import itertools
import csv
from collections import defaultdict


#get input array
def getInputArr( path ,path1 ,path2 , width , height):
	try:
		#read the image
		img = cv2.imread(path, 1)
		#resize it 
		img = cv2.resize(img, ( width , height ))
		#input must be float type
		img = img.astype(np.float32)

		#read the image
		img1 = cv2.imread(path1, 1)
		#resize it 
		img1 = cv2.resize(img1, ( width , height ))
		#input must be float type
		img1 = img1.astype(np.float32)

		#read the image
		img2 = cv2.imread(path2, 1)
		#resize it 
		img2 = cv2.resize(img2, ( width , height ))
		#input must be float type
		img2 = img2.astype(np.float32)

		#combine three imgs to  (width , height, rgb*3)
		imgs =  np.concatenate((img, img1, img2),axis=2)

		#since the odering of TrackNet  is 'channels_first', so we need to change the axis
		imgs = np.rollaxis(imgs, 2, 0)
		return imgs

	except Exception as e:

		print(path , e)



#get output array
def getOutputArr( path , nClasses ,  width , height, resize=1):

	# seg_labels = np.zeros((  height , width  , nClasses ))
	try:
		img = cv2.imread(path, 1)
		if resize:
			img = cv2.resize(img, ( width , height ))
		# img = img[:, : , 0]
		img = img[:, :, 0]/255
		img = img.astype(np.float32)

		# for c in range(nClasses):
		# 	seg_labels[: , : , c ] = (img == c ).astype(int)z

	except Exception as e:
		print(e)
		
	# seg_labels = np.reshape(seg_labels, ( width*height , nClasses ))
	# return seg_labels
	return img



#read input data and output data
def InputOutputGenerator( images_path,  batch_size,  n_classes , input_height , input_width , output_height , output_width ):


	#read csv file to 'zipped'
	columns = defaultdict(list)
	with open(images_path) as f:
		reader = csv.reader(f)
		next(reader)
		for row in reader:
			for (i,v) in enumerate(row):
				columns[i].append(v)
	zipped = itertools.cycle( zip(columns[0], columns[1], columns[2], columns[3]) )

	while True:
		Input = []
		Output = []
		#read input&output for each batch
		for _ in range( batch_size):
			path, path1, path2 , anno = next(zipped)
			Input.append( getInputArr(path, path1, path2 , input_width , input_height) )
			Output.append( getOutputArr(anno, n_classes , output_width , output_height) )
		#return input&output
		yield np.array(Input) , np.array(Output)


import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class TennisDataset(Dataset):
	def __init__(self, images_path,  n_classes , input_height , input_width , output_height , output_width, train=True, num_images=0, rt_ori=False):
		self.images_path = images_path
		self.n_classes = n_classes
		self.input_height = input_height
		self.input_width = input_width
		self.output_height = output_height
		self.output_width = output_width
		self.train = train
		self.resize = train
		self.rt_ori = rt_ori
		# read csv file to 'zipped'
		columns = defaultdict(list)
		with open(images_path) as f:
			reader = csv.reader(f)
			next(reader)
			for row in reader:
				for (i, v) in enumerate(row):
					columns[i].append(v)
		# if self.train:
		# 	with open('data/csv/mine_train.csv') as f:
		# 		reader = csv.reader(f)
		# 		next(reader)
		# 		for row in reader:
		# 			for (i, v) in enumerate(row):
		# 				columns[i].append(v)
		zipped = itertools.cycle(zip(columns[0], columns[1], columns[2], columns[3]))
		self.data = [next(zipped) for i in range(len(columns[0]))]
		# self.data = self.data[16000:]

		if num_images > 0:
			# select a random subset of the dataset
			rand = np.random.randint(0, len(self.data), size=(num_images))
			self.data = np.array(self.data)[rand]

	def __len__(self):
		return len(self.data)
	def __getitem__(self, index):
		path, path1, path2, anno = self.data[index]
		input = getInputArr(path, path1, path2, self.input_width, self.input_height)
		output = getOutputArr(anno , self.n_classes , self.output_width , self.output_height, self.resize)
		gt_ht = cv2.imread(anno, 1)
		gt_ht = cv2.resize(gt_ht, (self.input_width, self.input_height))
		if self.rt_ori:
			img = cv2.imread(path, 1)
			img1 = cv2.imread(path1, 1)
			img2 = cv2.imread(path2, 1)
			imgs = np.concatenate((img, img1, img2), axis=2)
			return np.array(input), np.array(output), np.array(gt_ht), np.array(imgs)
		return np.array(input), np.array(output), np.array(gt_ht)
		# return np.array(input), np.array(output), np.array(gt_ht), str(path)

import glob
class VideoDataset(Dataset):
	def __init__(self, video_path,  n_classes , input_height , input_width , output_height , output_width, train=True, num_images=0, rt_ori=False):
		self.images_path = video_path
		self.n_classes = n_classes
		self.input_height = input_height
		self.input_width = input_width
		self.output_height = output_height
		self.output_width = output_width
		self.train = train
		self.resize = train
		self.rt_ori = rt_ori
		# read csv file to 'zipped'
		# self.data = self.get_video_frames(video_path)
		video_name = os.path.basename(video_path)[:-4]
		self.data = glob.glob(f'tmp/{video_name}/imgs/*.png')
		self.data = sorted(self.data)

		if num_images > 0:
			# select a random subset of the dataset
			rand = np.random.randint(0, len(self.data), size=(num_images))
			self.data = np.array(self.data)[rand]

	def __len__(self):
		return len(self.data)
	def __getitem__(self, index):
		if index<2:
			return 1
		img = cv2.imread(self.data[index], 1)
		img1 = cv2.imread(self.data[index-1], 1)
		img2 = cv2.imread(self.data[index-2], 1)
		if self.rt_ori:
			imgs_ori = np.concatenate((img, img1, img2), axis=2)
		img = cv2.resize(img, (self.input_width, self.input_height))
		img1 = cv2.resize(img1, (self.input_width, self.input_height))
		img2 = cv2.resize(img2, (self.input_width, self.input_height))
		img = img.astype(np.float32)
		img1 = img1.astype(np.float32)
		img2 = img2.astype(np.float32)
		imgs = np.concatenate((img, img1, img2), axis=2)
		input = np.rollaxis(imgs, 2, 0)
		if self.rt_ori:
			return np.array(input), np.array(imgs_ori)

		return np.array(input)

	def get_video_frames(self, video_path):
		frames = []
		cap = cv2.VideoCapture(video_path)
		total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

		frame_indices = range(total_frames)

		for idx in frame_indices:
			cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
			ret, frame = cap.read()
			frames.append(frame)

		cap.release()
		return frames
