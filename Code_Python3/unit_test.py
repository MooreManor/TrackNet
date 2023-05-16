import tensorflow as tf
# y_true = [[[0.,1.]]]
y_true = [[[0.,0.9]]]
y_pred = [[[0.4,0.6]]]# 假设已经经过了softmax，所以和必须为1
loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
print(loss.numpy())

import torch

loss = -(0 * torch.log(torch.tensor(0.4)) + 0.9 * torch.log(torch.tensor(0.6)))
# loss = -(0 * torch.log(torch.tensor(0.4)) + 1.0 * torch.log(torch.tensor(0.6)))
print('torch.loss', loss)


def pt_categorical_crossentropy(pred, label):
    """
    使用pytorch 来实现 categorical_crossentropy
    """
    # print(-label * torch.log(pred))
    return torch.sum(-label * torch.log(pred))


loss = pt_categorical_crossentropy(torch.tensor(y_pred), torch.tensor(y_true))
print(loss)