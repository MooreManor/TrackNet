import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

def classify_metrics(pred, gt):
    seqlen = gt.shape[0]
    TP = 0
    ALL_HAS = 0
    FP = 0
    diff = 0
    for j in range(seqlen):
        if gt[j] == 1:
            start = max(0, j - 12)
            end = min(seqlen, j + 12)
            ALL_HAS += 1
            if 1 in pred[start:end]:
                TP += 1
                ind = start+np.where(pred[start:end]==1)[0][0]
                diff += abs(ind-j)
        if pred[j] == 1:
            start = max(0, j - 12)
            end = min(seqlen, j + 12)
            if 1 not in gt[start:end]:
                FP += 1
    return TP, ALL_HAS, FP, diff

from keras.losses import Loss

# class SmoothL1Loss(Loss):
#     def __init__(self, delta=1.0, **kwargs):
#         self.delta = delta
#         super().__init__(**kwargs)
#
#     def call(self, y_true, y_pred):
#         x = y_true - y_pred
#         abs_x = K.abs(x)
#         smooth_x = 0.5 * (x ** 2) * K.cast(abs_x < self.delta, 'float32') \
#                    + (abs_x - 0.5 * self.delta) * K.cast(abs_x >= self.delta, 'float32')
#         return K.mean(smooth_x, axis=-1)

def tennis_loss(y_true, y_pred):
    gt_indices = tf.where(K.equal(y_true, 1))
    gt_indices = tf.cast(gt_indices, dtype=tf.int32)
    seqlen = len(y_pred)
    seqlen = tf.cast(seqlen, dtype=tf.int32)
    # 初始化损失
    loss = 0.0

    # 遍历每个gt帧的索引
    for idx in gt_indices:
        # 获取当前gt帧的索引值
        idx_value = idx[0]

        # 计算正负12帧内的预测值
        start_idx = tf.maximum(idx_value - 12, 0)
        end_idx = tf.minimum(idx_value + 13, seqlen)
        pred_values = y_pred[start_idx:end_idx]
        # 统计正负12帧内1的数量
        num_ones = tf.reduce_sum(tf.cast(K.equal(pred_values, 1), tf.float32))

        # 如果有一个1，则损失为0
        if num_ones == 1:
            loss += 0.0
        elif num_ones == 0:
            loss += 5.0
        else:
            loss += num_ones-1

    # 返回平均损失
    print(tf.shape(y_true))
    return loss / tf.cast(tf.shape(gt_indices)[0], dtype=tf.float32)


if __name__ == '__main__':
    y_true = tf.constant([0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0], dtype=tf.float32)
    y_pred = tf.constant([0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0], dtype=tf.float32)
    print(tf.shape(y_true))
    res = tennis_loss(y_true, y_pred)