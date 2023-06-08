import argparse
import Models , LoadBatches
from keras import optimizers
# from tensorflow import keras
# from keras.utils import plot_model
from keras.utils.vis_utils import plot_model
import tensorflow as tf
# tf.compat.v1.disable_v2_behavior()
import numpy as np
from torchvision.utils import make_grid
import os
import cv2
from keras import backend as K

import resource
from keras.callbacks import LambdaCallback, Callback
class MemoryCallback(Callback):
    def on_epoch_end(self, epoch, log={}):
        print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

import tensorflow
import gc

# Reset Keras Session
def reset_keras():
    K.clear_session()
    tf.compat.v1.reset_default_graph()

    gc.collect()


def train_summaries(vis, epoch, step):
    rend_imgs = []
    for i in range(train_batch_size):
        vis_input = vis['vis_input'][i]
        vis_pred = vis['vis_pred'][i]
        # vis_output = vis['vis_output'][i]
        rend_imgs.append(torch.from_numpy(vis_input).permute(2, 0, 1))
        rend_imgs.append(torch.from_numpy(vis_pred).permute(2, 0, 1))
        # rend_imgs.append(torch.from_numpy(vis_output).permute(2, 0, 1))
    images_pred = make_grid(rend_imgs, nrow=2)
    images_pred = images_pred.numpy().transpose(1, 2, 0)
    save_dir = os.path.join('logs', 'train_keras_output_images')
    os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(
        os.path.join(save_dir, f'result_epoch{epoch:02d}_step{step:05d}.png'),
        cv2.cvtColor(images_pred, cv2.COLOR_BGR2RGB)
    )

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
criterion = tf.keras.losses.CategoricalCrossentropy()
#load TrackNet model
modelTN = Models.TrackNet.TrackNet
m = modelTN(n_classes, input_height=input_height, input_width=input_width)
m.compile(loss='categorical_crossentropy', optimizer= optimizer_name, metrics=['accuracy'])
# m.compile(loss='categorical_crossentropy', optimizer= optimizer_name, metrics=['categorical_accuracy'])

#check if need to retrain the model weights
if load_weights != "-1":
    # m.load_weights("weights/model." + load_weights, by_name=True)
    m.load_weights("weights/model." + load_weights)

# #show TrackNet details, save it as TrackNet.png
# plot_model( m , show_shapes=True , to_file='TrackNet.png')

#get TrackNet output height and width
model_output_height = m.outputHeight
model_output_width = m.outputWidth

#creat input data and output data
Generator = LoadBatches.InputOutputGenerator( training_images_name,  train_batch_size,  n_classes , input_height , input_width , model_output_height , model_output_width)
# Generator = LoadBatches.TennisDataset_keras(training_images_name, train_batch_size,  n_classes, input_height, input_width, model_output_height, model_output_width,  shuffle=True)


#start to train the model, and save weights until finish 
'''
m.fit_generator( Generator, step_per_epochs, epochs )
m.save_weights( save_weights_path + ".0" )

'''
import torch
import torch.nn.functional as F

#start to train the model, and save weights per 50 epochs

for ep in range(1, epochs+1 ):
    print("Epoch :", str(ep) + "/" + str(epochs))
    # input = next(Generator)[0]
    # label = next(Generator)[1]
    # pred = m.predict(input)  # 输出数据，shape 为 (1, 1)
    # loss = tf.keras.losses.categorical_crossentropy(label, pred)
    # mean_loss = tf.reduce_mean(loss)
    m.fit_generator(Generator, step_per_epochs, callbacks=[MemoryCallback()])
    # # pred = torch.from_numpy(pred)
    # # label = torch.from_numpy(label)
    # # F.cross_entropy(pred.reshape(-1, pred.shape[2]), torch.argmax(label, dim=2).reshape(-1))
    # # res = pt_categorical_crossentropy(pred, label)
    # if ep % 50 == 0:
    K.clear_session()
    tf.compat.v1.reset_default_graph()
    gc.collect()
    if ep % 20 == 0:
      m.save_weights(save_weights_path + ".0")

    #------------------------------------------------------
    # for step in range(step_per_epochs):
    # 	print("Step :", str(step) + "/" + str(step_per_epochs))
    # 	input, label = next(Generator)
    # 	with tf.GradientTape() as tape:
    # 		pred = m(input, training=True)
    # 		loss = criterion(label, pred)
    # 		mean_loss = tf.reduce_mean(loss)
    # 		print("Loss :", str(mean_loss.numpy()))
    # 	gradients = tape.gradient(mean_loss, m.trainable_variables)
    # 	optimizer_name.apply_gradients(zip(gradients, m.trainable_variables))
    # 	if step % 1 == 0:
    # 		vis_input = torch.from_numpy(input)
    # 		vis_input = vis_input.permute(0, 2, 3, 1)[:, :, :, 0:3].numpy().astype(np.uint8)
    # 		vis_pred = torch.from_numpy(pred.numpy()).reshape(train_batch_size, input_height, input_width, n_classes)
    # 		vis_pred = torch.argmax(vis_pred, dim=-1).unsqueeze(-1).repeat(1, 1, 1, 3).numpy().astype(np.uint8)
    # 		# vis_output = np.array(vis_output).astype(np.uint8)
    # 		vis = {'vis_input': vis_input,
    # 			   'vis_pred': vis_pred,}
    # 		train_summaries(vis, epoch=ep, step=step)


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




