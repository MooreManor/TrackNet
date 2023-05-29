# from keras.models import *
# from keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import math
def TrackNet( n_classes ,  input_height, input_width ): # input_height = 360, input_width = 640

	imgs_input = Input(shape=(9,input_height,input_width))

	#layer1
	x = Conv2D(64, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(imgs_input)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer2
	x = Conv2D(64, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer3
	x = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first' )(x)

	#layer4
	x = Conv2D(128, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer5
	x = Conv2D(128, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer6
	x = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first' )(x)

	#layer7
	x = Conv2D(256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer8
	x = Conv2D(256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer9
	x = Conv2D(256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer10
	x = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first' )(x)

	#layer11
	x = ( Conv2D(512, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer12
	x = ( Conv2D(512, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer13
	x = ( Conv2D(512, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer14
	x = ( UpSampling2D( (2,2), data_format='channels_first'))(x)

	#layer15
	x = ( Conv2D( 256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer16
	x = ( Conv2D( 256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer17
	x = ( Conv2D( 256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer18
	x = ( UpSampling2D( (2,2), data_format='channels_first'))(x)

	#layer19
	x = ( Conv2D( 128 , (3, 3), kernel_initializer='random_uniform', padding='same' , data_format='channels_first' ))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer20
	x = ( Conv2D( 128 , (3, 3), kernel_initializer='random_uniform', padding='same' , data_format='channels_first' ))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer21
	x = ( UpSampling2D( (2,2), data_format='channels_first'))(x)

	#layer22
	x = ( Conv2D( 64 , (3, 3), kernel_initializer='random_uniform', padding='same'  , data_format='channels_first' ))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer23
	x = ( Conv2D( 64 , (3, 3), kernel_initializer='random_uniform', padding='same'  , data_format='channels_first' ))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer24
	x =  Conv2D( n_classes , (3, 3) , kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	o_shape = Model(imgs_input , x ).output_shape
	print ("layer24 output shape:", o_shape[1],o_shape[2],o_shape[3])
	#layer24 output shape: 256, 360, 640

	OutputHeight = o_shape[2]
	OutputWidth = o_shape[3]

	#reshape the size to (256, 360*640)
	x = (Reshape((  -1  , OutputHeight*OutputWidth   )))(x)

	#change dimension order to (360*640, 256)
	x = (Permute((2, 1)))(x)

	#layer25
	gaussian_output = (Activation('softmax'))(x)

	model = Model( imgs_input , gaussian_output)
	model.outputWidth = OutputWidth
	model.outputHeight = OutputHeight

	#show model's details
	model.summary()

	return model


import torch
import torch.nn as nn

import torch.nn as nn

def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight,std=0.001)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight,std=0.001)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)
        nn.init.constant_(m.bias,0)
class TrackNet_pt(nn.Module):
	def __init__(self, n_classes, input_height, input_width):
		super(TrackNet_pt, self).__init__()
		self.n_classes = n_classes
		self.input_height = input_height
		self.input_width = input_width

		self.layer1 = nn.Conv2d(9, 64, kernel_size=3, padding=1)
		self.relu1 = nn.ReLU()
		self.bn1 = nn.BatchNorm2d(64)

		self.layer2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
		self.relu2 = nn.ReLU()
		self.bn2 = nn.BatchNorm2d(64)

		self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

		self.layer4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
		self.relu4 = nn.ReLU()
		self.bn4 = nn.BatchNorm2d(128)

		self.layer5 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
		self.relu5 = nn.ReLU()
		self.bn5 = nn.BatchNorm2d(128)

		self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

		self.layer7 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
		self.relu7 = nn.ReLU()
		self.bn7 = nn.BatchNorm2d(256)

		self.layer8 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
		self.relu8 = nn.ReLU()
		self.bn8 = nn.BatchNorm2d(256)

		self.layer9 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
		self.relu9 = nn.ReLU()
		self.bn9 = nn.BatchNorm2d(256)

		self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

		self.layer11 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
		self.relu11 = nn.ReLU()
		self.bn11 = nn.BatchNorm2d(512)

		self.layer12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.relu12 = nn.ReLU()
		self.bn12 = nn.BatchNorm2d(512)

		self.layer13 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.relu13 = nn.ReLU()
		self.bn13 = nn.BatchNorm2d(512)

		self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')

		self.layer15 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
		self.relu15 = nn.ReLU()
		self.bn15 = nn.BatchNorm2d(256)

		self.layer16 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
		self.relu16 = nn.ReLU()
		self.bn16 = nn.BatchNorm2d(256)

		self.layer17 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
		self.relu17 = nn.ReLU()
		self.bn17 = nn.BatchNorm2d(256)

		self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')

		self.layer19 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
		self.relu19 = nn.ReLU()
		self.bn19 = nn.BatchNorm2d(128)

		self.layer20 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
		self.relu20 = nn.ReLU()
		self.bn20 = nn.BatchNorm2d(128)

		self.upsample3 = nn.Upsample(scale_factor=2, mode='nearest')

		self.layer22 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
		self.relu22 = nn.ReLU()
		self.bn22 = nn.BatchNorm2d(64)

		self.layer23 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
		self.relu23 = nn.ReLU()
		self.bn23 = nn.BatchNorm2d(64)

		self.layer24 = nn.Conv2d(64, n_classes, kernel_size=3, padding=1)
		self.relu24 = nn.ReLU()
		self.bn24 = nn.BatchNorm2d(n_classes)

		# self.softmax = nn.Softmax(dim=1)
		# self.softmax = nn.Softmax(dim=2)

		self.outputWidth = None
		self.outputHeight = None
		self.init_weights()

	def forward(self, x):
		batch_size = x.shape[0]
		# layer1
		x = self.layer1(x)
		x = self.relu1(x)
		x = self.bn1(x)

		# layer2
		x = self.layer2(x)
		x = self.relu2(x)
		x = self.bn2(x)

		# layer3
		x = self.maxpool1(x)

		# layer4
		x = self.layer4(x)
		x = self.relu4(x)
		x = self.bn4(x)

		# layer5
		x = self.layer5(x)
		x = self.relu5(x)
		x = self.bn5(x)

		# layer6
		x = self.maxpool2(x)

		# layer7
		x = self.layer7(x)
		x = self.relu7(x)
		x = self.bn7(x)

		# layer8
		x = self.layer8(x)
		x = self.relu8(x)
		x = self.bn8(x)

		# layer9
		x = self.layer9(x)
		x = self.relu9(x)
		x = self.bn9(x)

		# layer10
		x = self.maxpool3(x)

		# layer11
		x = self.layer11(x)
		x = self.relu11(x)
		x = self.bn11(x)

		# layer12
		x = self.layer12(x)
		x = self.relu12(x)
		x = self.bn12(x)

		# layer13
		x = self.layer13(x)
		x = self.relu13(x)
		x = self.bn13(x)

		# layer14
		x = self.upsample1(x)

		# layer15
		x = self.layer15(x)
		x = self.relu15(x)
		x = self.bn15(x)

		# layer16
		x = self.layer16(x)
		x = self.relu16(x)
		x = self.bn16(x)

		# layer17
		x = self.layer17(x)
		x = self.relu17(x)
		x = self.bn17(x)

		# layer18
		x = self.upsample2(x)

		# layer19
		x = self.layer19(x)
		x = self.relu19(x)
		x = self.bn19(x)

		# layer20
		x = self.layer20(x)
		x = self.relu20(x)
		x = self.bn20(x)

		# layer21
		x = self.upsample3(x)

		# layer22
		x = self.layer22(x)
		x = self.relu22(x)
		x = self.bn22(x)

		# layer23
		x = self.layer23(x)
		x = self.relu23(x)
		x = self.bn23(x)

		# layer24
		x = self.layer24(x)
		x = self.relu24(x)
		x = self.bn24(x)

		# reshape the size to (n_classes, outputHeight*outputWidth)
		# 256, 360, 640
		# x = x.permute(0, 2, 3, 1)
		x = x.reshape(batch_size, -1, self.input_width * self.input_height)
		x = x.permute(0, 2, 1)

		# apply softmax to get the probability distribution
		# x = self.softmax(x)

		return x

	def init_weights(self, pretrained=''):
		# logger.info('=> init weights from normal distribution')
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				# nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				torch.nn.init.uniform_(m.weight, a=-0.05, b=0.05)
				# nn.init.normal_(m.weight, std=0.001)
				# n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				# m.weight.data.normal_(0, math.sqrt(2. / n))
				# for name, _ in m.named_parameters():
				# 	if name in ['bias']:
				# 		nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.ConvTranspose2d):
				nn.init.normal_(m.weight, std=0.001)
				for name, _ in m.named_parameters():
					if name in ['bias']:
						nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, std=0.01)
				nn.init.constant_(m.bias, 0)