import numpy as np

IMG_NORM_MEAN = [0.485, 0.456, 0.406]
IMG_NORM_STD = [0.229, 0.224, 0.225]
mean = np.array(IMG_NORM_MEAN, dtype=np.float32)
std = np.array(IMG_NORM_STD, dtype=np.float32)

