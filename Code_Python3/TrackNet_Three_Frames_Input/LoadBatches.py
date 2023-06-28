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
			Output.append( getOutputArr(anno, n_classes , output_width , output_height) )
		#return input&output
		yield np.array(Input) , np.array(Output)


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