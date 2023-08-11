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


import glob
from utils.utils import calculate_velocity, add_csv_col, jud_dir, add_text_to_video, interpolation
from utils.feat_utils import get_lag_feature, get_single_lag_feature
class TSCDataset(Dataset):
	def __init__(self, train=True, target_name = 'first_hit', var_list = ['x', 'y'], lag_num = 40, test_game = ['game9', 'game10']):
		csv_file_all = glob.glob('/datasetb/tennis/' + '/**/Label.csv', recursive=True)
		pre_lag = 9
		num_frames_from_event = 4
		aft_lag = lag_num - pre_lag - 1
		var_num = len(var_list)
		self.var_num = var_num
		self.x = np.empty((0, lag_num, var_num))
		self.y = np.empty((0,))
		self.match = []

		# target_name = 'bounce'
		csv_val = 1 if 'hit' in target_name else 2
		first = 1 if 'first' in target_name else 0
		last = 1 if 'last' in target_name else 0

		for csv_path in csv_file_all:
			data = []
			seq_X = []
			is_test_csv=False
			if any(x in csv_path for x in test_game):
				is_test_csv = True
			if train != (not is_test_csv):
				continue

			# if train == ~is_test_csv:
			if first:
				flag = 0
			with open(csv_path, newline='', encoding='gbk') as csvfile:
				reader = csv.reader(csvfile)
				next(reader)  # 跳过第一行
				for row in reader:
					bounce = 0
					if row[2] == '':
						data.append([None, None, 0])
					else:
						x, y = float(row[2]), float(row[3])
						# if int(row[4])==2:
						if int(row[4]) == csv_val:
							if first:
								if flag == 0:
									bounce = 1
									flag = 1
							else:
								bounce = 1
						else:
							bounce = 0
						data.append([x, y, bounce])
						self.match.append(csv_path)
			bounce = list(np.array(data)[:, 2])
			bounce = np.array([int(x) for x in bounce])
			# SMOOTH
			bo_indices = np.where(bounce == 1)
			if len(bo_indices[0]) != 0:
				for smooth_idx in bo_indices:
					# sub_smooth_frame_indices = [idx for idx in range(smooth_idx[0] - num_frames_from_event,
					#                                                  smooth_idx[0] + num_frames_from_event + 1)]
					sub_smooth_frame_indices = []
					for idx in range(smooth_idx[0] - num_frames_from_event,
									 smooth_idx[0] + num_frames_from_event + 1):
						if 0 <= idx < len(data):
							sub_smooth_frame_indices.append(idx)
					bounce[sub_smooth_frame_indices] = 1

			xy = [l[:2] for l in data]
			xy = interpolation(xy)
			x = list(np.array(xy)[:, 0])
			y = list(np.array(xy)[:, 1])
			v = np.diff(xy, axis=0)
			v = np.pad(v, ((1, 0), (0, 0)), 'constant', constant_values=0)
			vx = v[:, 0]
			vy = v[:, 1]
			a = np.diff(v, axis=0)
			a = np.pad(a, ((1, 0), (0, 0)), 'constant', constant_values=0)
			ax = a[:, 0]
			ay = a[:, 1]
			v = pow(pow(v[:, 0], 2) + pow(v[:, 1], 2), 0.5)
			a = pow(pow(a[:, 0], 2) + pow(a[:, 1], 2), 0.5)
			# seq_X = get_lag_feature(x, y, v)
			# tmp = get_single_lag_feature(x)
			for var in var_list:
				tmp = get_single_lag_feature(eval(var), pre_lag=pre_lag, aft_lag=aft_lag)
				seq_X.append(tmp)
			seq_X = np.concatenate(seq_X, axis=2)
			# seq_Y = bounce
			seq_Y = bounce[9:-31]

			self.x = np.concatenate([self.x, seq_X], axis=0)
			self.y = np.concatenate([self.y, seq_Y], axis=0)

		# self.x_pos =
		# self.x_neg =

	def __len__(self):
		return len(self.x)

	def __getitem__(self, index):
		input = self.x[index]
		label = self.y[index]
		if np.random.uniform() <= 0.5:
		    lower_bound = -5
		    upper_bound = 5
		    mean = (lower_bound + upper_bound) / 2
		    std = (upper_bound - lower_bound) / 6
		    noise = np.random.normal(mean, std, size=(40, self.var_num))
		    # noise = np.random.uniform(low=-1.0, high=1.0, size=(y_train.shape[0], lag_num, var_num))*5
		    input = input + noise
		if np.random.uniform() <= 0.5:
		    input[:, 0] = 1280-input[:, 0]
		return input, label