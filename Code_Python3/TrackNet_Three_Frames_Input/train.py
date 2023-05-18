import argparse
import Models , LoadBatches
from keras import optimizers
# from tensorflow import keras
# from keras.utils import plot_model
from keras.utils.vis_utils import plot_model
import tensorflow as tf

# --save_weights_path=weights/model --training_images_name="training_model3_mine.csv" --epochs=500 --n_classes=256 --input_height=360 --input_width=640 --load_weights=2 --step_per_epochs=200 --batch_size=2
# --save_weights_path=weights/model --training_images_name="training_model2.csv" --epochs=500 --n_classes=256 --input_height=360 --input_width=640 --load_weights=2 --step_per_epochs=200 --batch_size=2
#parse parameters
parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path", type = str  )
parser.add_argument("--training_images_name", type = str  )
parser.add_argument("--n_classes", type=int )
parser.add_argument("--input_height", type=int , default = 360  )
parser.add_argument("--input_width", type=int , default = 640 )
parser.add_argument("--epochs", type = int, default = 1000 )
parser.add_argument("--batch_size", type = int, default = 2 )
parser.add_argument("--load_weights", type = str , default = "-1")
parser.add_argument("--step_per_epochs", type = int, default = 200 )

args = parser.parse_args()
training_images_name = args.training_images_name
train_batch_size = args.batch_size
n_classes = args.n_classes
input_height = args.input_height
input_width = args.input_width
save_weights_path = args.save_weights_path
epochs = args.epochs
load_weights = args.load_weights
step_per_epochs = args.step_per_epochs
# optimizer_name = optimizers.Adadelta(lr=1.0)
# optimizer_name = optimizers.Adam(lr=1.0)
from tensorflow.keras.optimizers import Adadelta, Adam
# from keras.optimizers import Adadelta
optimizer_name = Adadelta(lr=1.0)
# optimizer_name = tf.python.keras.optimizers.Adadelta(lr=1.0)
# optimizer_name = tf.python.keras.optimizers.Adadelta(lr=1.0)
# tensorflow.python.keras.optimizers

#load TrackNet model
modelTN = Models.TrackNet.TrackNet
m = modelTN( n_classes , input_height=input_height, input_width=input_width   )
m.compile(loss='categorical_crossentropy', optimizer= optimizer_name, metrics=['accuracy'])
# m.compile(loss='categorical_crossentropy', optimizer= optimizer_name, metrics=['categorical_accuracy'])

#check if need to retrain the model weights
if load_weights != "-1":
	m.load_weights("weights/model." + load_weights)

# #show TrackNet details, save it as TrackNet.png
# plot_model( m , show_shapes=True , to_file='TrackNet.png')

#get TrackNet output height and width
model_output_height = m.outputHeight
model_output_width = m.outputWidth

#creat input data and output data
Generator = LoadBatches.InputOutputGenerator( training_images_name,  train_batch_size,  n_classes , input_height , input_width , model_output_height , model_output_width)


#start to train the model, and save weights until finish 
'''
m.fit_generator( Generator, step_per_epochs, epochs )
m.save_weights( save_weights_path + ".0" )

'''

#start to train the model, and save weights per 50 epochs  

for ep in range(1, epochs+1 ):
	print("Epoch :", str(ep) + "/" + str(epochs))
	m.fit_generator(Generator, step_per_epochs)
	if ep % 50 == 0:
		m.save_weights(save_weights_path + ".0")


# # Pytorch Implementation
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torchvision.transforms import ToTensor
# from tqdm import tqdm
#
# def pt_categorical_crossentropy(pred, label):
#     """
#     使用pytorch 来实现 categorical_crossentropy
#     """
#     # print(-label * torch.log(pred))
#     return torch.sum(-label * torch.log(pred))
#
#
# model = Models.TrackNet.TrackNet(n_classes, input_height, input_width)
# optimizer = optim.Adadelta(model.parameters(), lr=1.0)
# # criterion = nn.CrossEntropyLoss()
# # 获取TrackNet输出的高度和宽度
# model_output_height = 360
# model_output_width = 640




