# -*- coding: utf-8 -*-
# @Time    : 2020/12/2 17:37
# @Author  : cmdxmm
# @FileName: test.py
# @Email   ：lidongyang@mail.sdu.edu.cn

from network.Model_define import get_custom_objects,Network_LSTM_MSCNet
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from dataset_dong import dataset
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
my_seed = 666
np.random.seed(my_seed)
random.seed(my_seed)
tf.random.set_seed(my_seed)

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.train_losses = []
        self.val_losses = []
    def on_epoch_end(self, batch, logs={}):
        self.train_losses.append(logs.get('accuracy'))
        self.val_losses.append(logs.get('val_accuracy'))

input1 = keras.Input(shape=(4096,), name='input1')
input2 = keras.Input(shape=(4096,), name='input2')
input3 = keras.Input(shape=(4096,), name='input3')
output = Network_LSTM_MSCNet(input1, input2, input3)
model = keras.Model(inputs=[input1, input2, input3], outputs=output, name="output")
_custom_objects = get_custom_objects()  # load keywords of Custom layers
model.summary()

adam = keras.optimizers.Adam(learning_rate=0.0006)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
filepath = "./model/weightsbestfinal_LSTM_MSCNet.h5"
checkpoints = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
history = LossHistory()
callbacks_list = [checkpoints,history]
datapath = 'newtrainData3_3_30'
x_train,  x_test, y_train , y_test = dataset(datapath)

model.fit([x_train[:, 0, :], x_train[:, 1, :], x_train[:, 2, :], ], to_categorical(y_train),
          validation_data=([x_test[:, 0, :], x_test[:, 1, :], x_test[:, 2, :]], to_categorical(y_test)), epochs=100,
          batch_size=512, callbacks=callbacks_list)
print(history.train_losses)
print(history.val_losses)
np.save('./loss/train_losses_LSTM_MSCNet.npy', np.array(history.train_losses))
np.save('./loss/val_losses_LSTM_MSCNet.npy', np.array(history.val_losses))
