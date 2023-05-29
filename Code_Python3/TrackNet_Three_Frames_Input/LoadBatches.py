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
def getOutputArr( path , nClasses ,  width , height  ):

	seg_labels = np.zeros((  height , width  , nClasses ))
	try:
		img = cv2.imread(path, 1)
		img = cv2.resize(img, ( width , height ))
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
			path, path1, path2 , anno = next(zipped)
			Input.append( getInputArr(path, path1, path2 , input_width , input_height) )
			Output.append( getOutputArr( anno , n_classes , output_width , output_height) )
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