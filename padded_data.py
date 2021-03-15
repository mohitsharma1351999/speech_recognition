# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 09:26:22 2020

@author: ARSHDEEP SINGH
"""

import numpy as np 
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

full_data = np.load('dev.npy', allow_pickle=True, encoding='bytes')
full_data.shape
array1 = []
for array in full_data:
    array1.append(array.shape[0])
max_width = max(array1)

padded_arrays = []
for array in full_data:
    padding_value = max_width - array.shape[0]
    paded_mel = np.pad(array, ((0, padding_value), (0, 0)),'constant', constant_values=(0))
    padded_arrays.append(paded_mel)    

padded_arrays = np.array(padded_arrays)
print('done with padding', 'shape of the padded mel---->', padded_arrays.shape)

labels = np.load('dev_labels.npy', allow_pickle=True, encoding='bytes')
padded_labels = []
for label in labels:
    padding_value = max_width - label.shape[0]
    temp = []
    temp.extend(label)
    temp.extend([-1]*padding_value)
    padded_labels.append(temp)

padded_labels = np.array(padded_labels)
print('labels array shape: ',padded_labels.shape)


X_train, X_test, y_train, y_test = train_test_split(padded_arrays, padded_labels, test_size=0.3)
print("X_train", X_train.shape)
print("y_train", y_train.shape)
print("X_test", X_test.shape)
print("y_test", y_test.shape)

classes = np.unique(y_train)
classes_num = len(classes)
print('Total number of outputs : ', classes_num)
print('Output classes : ', classes)


# one hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
