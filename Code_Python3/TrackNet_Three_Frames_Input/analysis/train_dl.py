import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import sklearn
import numpy as np
from sklearn import preprocessing
import warnings
# 忽略所有警告
warnings.filterwarnings('ignore')
# 忽略特定类型的警告
warnings.filterwarnings('ignore', category=DeprecationWarning)
from utils.utils import calculate_velocity, add_csv_col, jud_dir, add_text_to_video, interpolation
from utils.metrics import classify_metrics, tennis_loss
from utils.feat_utils import get_lag_feature, get_single_lag_feature
import csv
import numpy as np


import glob

y_train = np.empty((0,))
y_test = np.empty((0,))

target_name = 'hit'
# target_name = 'first_hit'
# classifier_name = 'fcn'
# classifier_name = 'inception'
classifier_name = 'resnet'

lag_num = 20
var_list = ['x', 'y', 'vx', 'vy', 'ax', 'ay', 'v', 'a']
# var_list = ['x', 'y', 'v', 'a']
# var_list = ['x', 'y', 'vx', 'vy', 'ax', 'ay']
var_num = len(var_list)
x_test = np.empty((0, lag_num, var_num))
x_train = np.empty((0, lag_num, var_num))

# target_name = 'bounce'
csv_val = 1 if 'hit' in target_name else 2
first = 1 if 'first' in target_name else 0
last = 1 if 'last' in target_name else 0

test_game = ['game9', 'game10']
csv_file_all = glob.glob('/datasetb/tennis/' + '/**/Label.csv', recursive=True)
for csv_path in csv_file_all:
    data = []
    seq_X = []
    test = 0
    if any(x in csv_path for x in test_game):
        test = 1

    if first:
        flag=0
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

    bounce = list(np.array(data)[:, 2])
    bounce = np.array([int(x) for x in bounce])
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
        tmp = get_single_lag_feature(eval(var), lag=lag_num)
        seq_X.append(tmp)
    seq_X = np.concatenate(seq_X, axis=2)
    seq_Y = bounce
    if test==1:
        x_test = np.concatenate([x_test, seq_X], axis=0)
        y_test = np.concatenate([y_test, seq_Y], axis=0)
    else:
        x_train = np.concatenate([x_train, seq_X], axis=0)
        y_train = np.concatenate([y_train, seq_Y], axis=0)

def fit_classifier():
    # x_train = datasets_dict[dataset_name][0]
    # y_train = datasets_dict[dataset_name][1]
    # x_test = datasets_dict[dataset_name][2]
    # y_test = datasets_dict[dataset_name][3]
    # x_train = np.random.randn(1000, 10, 2)
    # y_train = np.random.randint(0, 2, size=1000)
    # x_test = np.random.randn(1000, 10, 2)
    # y_test = np.random.randint(0, 2, size=1000)
    global y_train, y_test, x_train, x_test
    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    # transform the labels from integers to one hot vectors
    # enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc = preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    # save orignal y because later we will use binary
    y_true = np.argmax(y_test, axis=1)

    # if len(x_train.shape) == 2:  # if univariate
    #     # add a dimension to make it multivariate with one dimension
    #     x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    #     x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    input_shape = x_train.shape[1:]
    classifier = create_classifier(classifier_name, input_shape, nb_classes, output_directory)

    TP, ALL_HAS, FP, diff, TP_tr, ALL_HAS_tr, FP_tr, diff_tr = classifier.fit(x_train, y_train, x_test, y_test, y_true)

    print(f'{target_name}_train结果')
    print('模型预测正确平均绝对差: ', diff_tr / TP_tr)
    print(f'模型预测正确个数/GT个数: {TP_tr}/{ALL_HAS_tr}')
    print('没有却检测出来个数: ', FP_tr)
    print(f'{target_name}_test结果')
    print('模型预测正确平均绝对差: ', diff / TP)
    print(f'模型预测正确个数/GT个数: {TP}/{ALL_HAS}')
    print('没有却检测出来个数: ', FP)

def create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose=True):
    if classifier_name == 'fcn':
        from classifiers import fcn
        return fcn.Classifier_FCN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mlp':
        from classifiers import mlp
        return mlp.Classifier_MLP(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'resnet':
        from classifiers import resnet
        return resnet.Classifier_RESNET(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mcnn':
        from classifiers import mcnn
        return mcnn.Classifier_MCNN(output_directory, verbose)
    if classifier_name == 'tlenet':
        from classifiers import tlenet
        return tlenet.Classifier_TLENET(output_directory, verbose)
    if classifier_name == 'twiesn':
        from classifiers import twiesn
        return twiesn.Classifier_TWIESN(output_directory, verbose)
    if classifier_name == 'encoder':
        from classifiers import encoder
        return encoder.Classifier_ENCODER(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mcdcnn':
        from classifiers import mcdcnn
        return mcdcnn.Classifier_MCDCNN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'cnn':  # Time-CNN
        from classifiers import cnn
        return cnn.Classifier_CNN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'inception':
        from classifiers import inception
        return inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose)


if __name__ == '__main__':
    root_dir = './'
    itr = ''
    output_directory = root_dir + '/results/' + classifier_name + '/' + target_name
    os.makedirs(output_directory, exist_ok=True)
    fit_classifier()