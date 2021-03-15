import numpy as np
from keras.models import Sequential
from keras.layers import Dense

arr = np.load('dev.npy', allow_pickle=True, encoding='bytes')
print(arr.shape)

arr_label=np.load("dev_labels.npy", allow_pickle=True, encoding='bytes')
# print(arr_label[0])

s=0
for i in range(0,arr.shape[0]):
    s = s+arr[i].shape[0]
print(s)

stack_arr=np.full([1,40],None)   
print(stack_arr) 
for i in range(0,arr.shape[0]):
    stack_arr=np.vstack((stack_arr,arr[i]))

stack_arr=stack_arr[1:s+1] 
print(stack_arr.shape)

stack_arr_label=np.full([1,1],None)    
for i in range(0,arr_label.shape[0]):
    stack_arr_label=np.vstack((stack_arr_label,arr_label[i].reshape(-1,1)))
print(stack_arr_label.shape)
stack_arr_label=stack_arr_label[1:s+1]
# stack_arr_label = np.delete(stack_arr_label,0,0)

final_label = []
for label in stack_arr_label:
    final_label.append(label[0])
final_label = np.array(final_label)
print(final_label.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(stack_arr,final_label, test_size=0.30, random_state=42)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape ) 
print(y_test.shape)
x_train = np.asarray(x_train).astype(np.float32)
y_train = np.asarray(y_train).astype(np.float32)



model = Sequential()
model.add(Dense(640, input_dim=40, activation='relu'))
model.add(Dense(320, activation='relu'))
model.add(Dense(138, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=20, batch_size=10)