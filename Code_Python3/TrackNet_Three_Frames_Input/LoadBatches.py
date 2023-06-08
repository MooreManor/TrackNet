import matplotlib.pyplot as plt
import numpy as np
import cv2
import itertools
import csv
from collections import defaultdict
from utils.constants import mean, std

# IMG_NORM_MEAN = [0.485, 0.456, 0.406]
# IMG_NORM_STD = [0.229, 0.224, 0.225]
# mean = np.array(IMG_NORM_MEAN, dtype=np.float32)
# std = np.array(IMG_NORM_STD, dtype=np.float32)

#get input array
def getInputArr( path ,path1 ,path2 , width , height, flip, rot, pn):
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

		# random flip
		if flip==1:
			img = cv2.flip(img, 1)
			img1 = cv2.flip(img1, 1)
			img2 = cv2.flip(img2, 1)

		# random rotation
		M = cv2.getRotationMatrix2D((width / 2, height / 2), rot, 1)
		img = cv2.warpAffine(img, M, (width, height))
		img1 = cv2.warpAffine(img1, M, (width, height))
		img2 = cv2.warpAffine(img2, M, (width, height))

		# in the rgb image we add pixel noise in a channel-wise manner
		img[:, :, 0] = np.minimum(255.0, np.maximum(0.0, img[:, :, 0] * pn[0]))
		img[:, :, 1] = np.minimum(255.0, np.maximum(0.0, img[:, :, 1] * pn[1]))
		img[:, :, 2] = np.minimum(255.0, np.maximum(0.0, img[:, :, 2] * pn[2]))
		img1[:, :, 0] = np.minimum(255.0, np.maximum(0.0, img1[:, :, 0] * pn[0]))
		img1[:, :, 1] = np.minimum(255.0, np.maximum(0.0, img1[:, :, 1] * pn[1]))
		img1[:, :, 2] = np.minimum(255.0, np.maximum(0.0, img1[:, :, 2] * pn[2]))
		img2[:, :, 0] = np.minimum(255.0, np.maximum(0.0, img2[:, :, 0] * pn[0]))
		img2[:, :, 1] = np.minimum(255.0, np.maximum(0.0, img2[:, :, 1] * pn[1]))
		img2[:, :, 2] = np.minimum(255.0, np.maximum(0.0, img2[:, :, 2] * pn[2]))
		#combine three imgs to  (width , height, rgb*3)
		# img = (img - mean) / std
		# img1 = (img1 - mean) / std
		# img2 = (img2 - mean) / std
		imgs = np.concatenate((img, img1, img2),axis=2)

		#since the odering of TrackNet  is 'channels_first', so we need to change the axis
		imgs = np.rollaxis(imgs, 2, 0)
		return imgs

	except Exception as e:

		print(path , e)



#get output array
def getOutputArr( path , nClasses ,  width , height , flip, rot):

	seg_labels = np.zeros((  height , width  , nClasses ))
	try:
		img = cv2.imread(path, 1)
		img = cv2.resize(img, ( width , height ))
		# random flip
		if flip == 1:
			img = cv2.flip(img, 1)

		# random rotation
		M = cv2.getRotationMatrix2D((width / 2, height / 2), rot, 1)
		img = cv2.warpAffine(img, M, (width, height))
		img = img[:, : , 0]

		for c in range(nClasses):
			seg_labels[: , : , c ] = (img == c ).astype(int)

	except Exception as e:
		print(e)
		
	seg_labels = np.reshape(seg_labels, ( width*height , nClasses ))
	return seg_labels



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
			flip = 0
			if np.random.uniform() <= 0.5:
				flip = 1
			rot = np.random.randint(-10, 10)
			pn = np.random.uniform(1 - 0.4, 1 + 0.4, 3)

			path, path1, path2 , anno = next(zipped)
			Input.append( getInputArr(path, path1, path2 , input_width , input_height, flip, rot, pn) )
			Output.append( getOutputArr( anno , n_classes , output_width , output_height, flip, rot) )
		#return input&output
		yield np.array(Input) , np.array(Output)

import keras
import tensorflow as tf
class TennisDataset_keras(tf.keras.utils.Sequence):
	def __init__(self, images_path,  batch_size, n_classes , input_height , input_width , output_height , output_width,  shuffle=True, num_images=0):
		self.images_path = images_path
		self.n_classes = n_classes
		self.input_height = input_height
		self.input_width = input_width
		self.output_height = output_height
		self.output_width = output_width
		self.batch_size = batch_size
		self.shuffle = shuffle
		# read csv file to 'zipped'
		columns = defaultdict(list)
		with open(images_path) as f:
			reader = csv.reader(f)
			next(reader)
			for row in reader:
				for (i, v) in enumerate(row):
					columns[i].append(v)
		zipped = itertools.cycle(zip(columns[0], columns[1], columns[2], columns[3]))
		self.data = [next(zipped) for i in range(len(columns[0]))]

		if num_images > 0:
			# select a random subset of the dataset
			rand = np.random.randint(0, len(self.data), size=(num_images))
			self.data = np.array(self.data)[rand]

	def __len__(self):
		# 计算每个 epoch 需要的迭代次数
		return len(self.data) // self.batch_size

	def __getitem__(self, idx):
		# 获取当前 batch 的图像数据和标签数据

		# 计算当前 batch 的起始索引和结束索引
		batch_start = idx * self.batch_size
		batch_end = (idx + 1) * self.batch_size

		# 读取当前 batch 的图像文件名和标签数据
		batch_data = self.data[batch_start:batch_end]
		batch_path = [item[0] for item in batch_data]
		batch_path1 = [item[1] for item in batch_data]
		batch_path2 = [item[2] for item in batch_data]
		batch_anno = [item[3] for item in batch_data]
		batch_images = []
		batch_labels = []
		for path, path1, path2, anno in zip(batch_path, batch_path1, batch_path2, batch_anno):
			# 读取图像文件并调整大小
			input = getInputArr(path, path1, path2, self.input_width, self.input_height)

			# 将图像数据加入 batch_images 列表中
			batch_images.append(input)

			# 根据文件名获取标签数据并将其加入 batch_labels 列表中
			# 这里假设文件名的格式为 "label_filename.jpg"
			output = getOutputArr(anno, self.n_classes , self.output_width , self.output_height)
			batch_labels.append(output)

		# 将图像数据和标签数据转换为 NumPy 数组并返回
		# return np.array(batch_images), np.array(batch_labels)
		return np.array(batch_images), np.array(batch_labels)

	def on_epoch_end(self):
		# 在每个 epoch 结束时打乱图像文件名的顺序
		if self.shuffle:
			np.random.shuffle(self.data)

import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class TennisDataset(Dataset):
	def __init__(self, images_path,  n_classes , input_height , input_width , output_height , output_width, num_images=0):
		self.images_path = images_path
		self.n_classes = n_classes
		self.input_height = input_height
		self.input_width = input_width
		self.output_height = output_height
		self.output_width = output_width
		# read csv file to 'zipped'
		columns = defaultdict(list)
		with open(images_path) as f:
			reader = csv.reader(f)
			next(reader)
			for row in reader:
				for (i, v) in enumerate(row):
					columns[i].append(v)
		zipped = itertools.cycle(zip(columns[0], columns[1], columns[2], columns[3]))
		self.data = [next(zipped) for i in range(len(columns[0]))]

		if num_images > 0:
			# select a random subset of the dataset
			rand = np.random.randint(0, len(self.data), size=(num_images))
			self.data = np.array(self.data)[rand]

	def __len__(self):
		return len(self.data)
	def __getitem__(self, index):
		path, path1, path2, anno = self.data[index]
		input = getInputArr(path, path1, path2, self.input_width, self.input_height)
		output = getOutputArr(anno , self.n_classes , self.output_width , self.output_height)
		gt_ht = cv2.imread(anno, 1)
		gt_ht = cv2.resize(gt_ht, (self.input_width, self.input_height))
		return np.array(input), np.array(output), np.array(gt_ht)