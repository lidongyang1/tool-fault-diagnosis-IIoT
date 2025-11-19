# -*- coding: utf-8 -*-
# @Time    : 2020/12/2 17:37
# @Author  : cmdxmm
# @FileName: test.py
# @Email   ：lidongyang@mail.sdu.edu.cn

from dataset_dong import dataset
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import random
import tensorflow as tf
from tensorflow import keras

my_seed = 666
np.random.seed(my_seed)
random.seed(my_seed)
tf.random.set_seed(my_seed)

datapath = 'newtrainData3_3_30'
x_train,  x_test, y_train , y_test = dataset(datapath)
model = keras.models.load_model("./model/weightsbestfinal_LSTM_MSCNet.h5")
prediction = np.argmax(model.predict([x_test[:, 0, :], x_test[:, 1, :], x_test[:, 2, :], ]),axis=1)
print(prediction)
print('Accuracy_score:',accuracy_score(y_test, prediction))
print('Precision_score:',precision_score(y_test, prediction,average='weighted'))
print('Recall_score:',recall_score(y_test, prediction,average='weighted'))
print('F1_score:',f1_score(y_test, prediction,average='weighted'))

