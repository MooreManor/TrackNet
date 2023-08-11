# resnet model
# when tuning start with learning rate->mini_batch_size ->
# momentum-> #hidden_units -> # learning_rate_decay -> #layers
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time

import matplotlib
# from utils.utils import save_test_duration
# from utils.metrics import tennis_loss


matplotlib.use('agg')
import matplotlib.pyplot as plt

# from utils.utils import save_logs
# from utils.utils import calculate_metrics


class Classifier_RESNET:

    def __init__(self, output_directory, input_shape, nb_classes, verbose=False, build=True, load_weights=False):
        self.output_directory = output_directory
        if build == True:
            self.model = self.build_model(input_shape, nb_classes)
            if (verbose == True):
                self.model.summary()
            self.verbose = verbose
            if load_weights == True:
                self.model.load_weights(self.output_directory
                                        .replace('resnet_augment', 'resnet')
                                        .replace('TSC_itr_augment_x_10', 'TSC_itr_10')
                                        + '/model_init.hdf5')
            else:
                self.model.save_weights(self.output_directory + 'model_init.hdf5')
        return

    def build_model(self, input_shape, nb_classes):
        n_feature_maps = 64

        input_layer = keras.layers.Input(input_shape)

        # BLOCK 1

        conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_1 = keras.layers.add([shortcut_y, conv_z])
        output_block_1 = keras.layers.Activation('relu')(output_block_1)

        # BLOCK 2

        # conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
        # conv_x = keras.layers.BatchNormalization()(conv_x)
        # conv_x = keras.layers.Activation('relu')(conv_x)
        #
        # conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        # conv_y = keras.layers.BatchNormalization()(conv_y)
        # conv_y = keras.layers.Activation('relu')(conv_y)
        #
        # conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        # conv_z = keras.layers.BatchNormalization()(conv_z)
        #
        # # expand channels for the sum
        # shortcut_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
        # shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
        #
        # output_block_2 = keras.layers.add([shortcut_y, conv_z])
        # output_block_2 = keras.layers.Activation('relu')(output_block_2)

        # # BLOCK 3
        #
        # conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
        # conv_x = keras.layers.BatchNormalization()(conv_x)
        # conv_x = keras.layers.Activation('relu')(conv_x)
        #
        # conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        # conv_y = keras.layers.BatchNormalization()(conv_y)
        # conv_y = keras.layers.Activation('relu')(conv_y)
        #
        # conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        # # conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(l=0.001), bias_regularizer=tf.keras.regularizers.l2(l=0.001))(conv_y)
        # conv_z = keras.layers.BatchNormalization()(conv_z)
        #
        # # no need to expand channels because they are equal
        # shortcut_y = keras.layers.BatchNormalization()(output_block_2)
        #
        # output_block_3 = keras.layers.add([shortcut_y, conv_z])
        # output_block_3 = keras.layers.Activation('relu')(output_block_3)

        # FINAL

        # gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)
        gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_1)

        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
        # model.compile(loss=tennis_loss, optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)

        file_path = self.output_directory + 'best_model.hdf5'

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
                                                           save_best_only=True)

        self.callbacks = [reduce_lr, model_checkpoint]

        return model

    def fit(self, x_train, y_train, x_val, y_val, y_true):
        if not tf.test.is_gpu_available:
            print('error')
            exit()
        # x_val and y_val are only used to monitor the test loss and NOT for training
        batch_size = 64
        # nb_epochs = 1500
        nb_epochs = 200

        mini_batch_size = int(min(x_train.shape[0] / 10, batch_size))

        start_time = time.time()

        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
                              verbose=self.verbose, validation_data=(x_val, y_val), callbacks=self.callbacks)

        duration = time.time() - start_time

        self.model.save(self.output_directory + 'last_model.hdf5')

        y_pred = self.predict(x_val, y_true, x_train, y_train, y_val,
                              return_df_metrics=False)
        y_pred = np.argmax(y_pred, axis=1)

        from utils.metrics import classify_metrics
        TP, ALL_HAS, FP, diff = classify_metrics(y_pred, y_true)

        y_true = np.argmax(y_train, axis=1)
        y_pred = self.predict(x_train, y_true, x_train, y_train, y_val,
                              return_df_metrics=False)
        y_pred = np.argmax(y_pred, axis=1)
        TP_tr, ALL_HAS_tr, FP_tr, diff_tr = classify_metrics(y_pred, y_true)

        keras.backend.clear_session()
        return TP, ALL_HAS, FP, diff, TP_tr, ALL_HAS_tr, FP_tr, diff_tr

    def predict(self, x_test, y_true, x_train, y_train, y_test, return_df_metrics=True):
        start_time = time.time()
        model_path = self.output_directory + 'best_model.hdf5'
        # model_path = self.output_directory + 'last_model.hdf5'
        model = keras.models.load_model(model_path)
        y_pred = model.predict(x_test)
        if return_df_metrics:
            y_pred = np.argmax(y_pred, axis=1)
            df_metrics = calculate_metrics(y_true, y_pred, 0.0)
            return df_metrics
        else:
            test_duration = time.time() - start_time
            # save_test_duration(self.output_directory + 'test_duration.csv', test_duration)
            return y_pred

import torch
import torch.nn as nn
import torch.nn.functional as F
class Classifier_RESNET_pt(nn.Module):
    def __init__(self, nb_classes, channel_num):
        super(Classifier_RESNET_pt, self).__init__()
        n_feature_maps = 64

        # BLOCK 1
        self.conv1 = nn.Conv1d(channel_num, n_feature_maps, kernel_size=8, padding=4)
        self.bn1 = nn.BatchNorm1d(n_feature_maps)
        self.relu = nn.ReLU()

        # conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
        # conv_x = keras.layers.BatchNormalization()(conv_x)
        # conv_x = keras.layers.Activation('relu')

        self.conv2 = nn.Conv1d(n_feature_maps, n_feature_maps, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(n_feature_maps)
        # conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
        # conv_y = keras.layers.BatchNormalization()(conv_y)
        # conv_y = keras.layers.Activation('relu')(conv_y)

        self.conv3 = nn.Conv1d(n_feature_maps, n_feature_maps, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(n_feature_maps)
        # conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
        # conv_z = keras.layers.BatchNormalization()(conv_z)

        self.conv4 = nn.Conv1d(n_feature_maps, n_feature_maps, kernel_size=1, padding=0)
        self.bn4 = nn.BatchNorm1d(n_feature_maps)
        # expand channels for the sum
        # shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
        # shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        # output_block_1 = keras.layers.add([shortcut_y, conv_z])
        # output_block_1 = keras.layers.Activation('relu')(output_block_1)

        # gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_1)
        # output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)
        self.gap_layer = nn.AdaptiveAvgPool1d(1)
        self.output_layer = nn.Linear(n_feature_maps, nb_classes)

    def forward(self, x):
        conv_x = self.conv1(x)
        conv_x = self.bn1(conv_x)
        conv_x = self.relu(conv_x)

        conv_y = self.conv2(conv_x)
        conv_y = self.bn2(conv_y)
        conv_y = self.relu(conv_y)

        conv_z = self.conv3(conv_y)
        conv_z = self.bn3(conv_z)

        shortcut_y = self.conv4(conv_z)
        shortcut_y = self.bn4(shortcut_y)

        output_block_1 = shortcut_y+conv_z
        output_block_1 = self.relu(output_block_1)

        gap_layer = self.gap_layer(output_block_1)
        gap_layer = gap_layer.view(gap_layer.size(0), -1)

        output_layer = self.output_layer(gap_layer)
        output_layer = F.softmax(output_layer, dim=1)

        return output_layer

if __name__ == '__main__':
    inp = torch.randn(2, 2, 40)
    net = Classifier_RESNET_pt(2, 2)
    res = net(inp)








