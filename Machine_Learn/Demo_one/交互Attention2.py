from __future__ import print_function

import csv

from keras.utils import np_utils

import keras
from keras.datasets import mnist
from keras.datasets.mnist import load_data
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, LSTM, Flatten, Conv1D, MaxPooling1D, Dense, Activation, \
    Dropout, GlobalMaxPooling1D, AveragePooling2D, ConvLSTM2D, GlobalMaxPooling2D, Recurrent, Reshape, Bidirectional, \
    BatchNormalization, Merge, concatenate, Permute, Lambda, RepeatVector, merge, add
from keras.utils import  to_categorical
from keras.layers import Embedding
from pandas.core.frame import DataFrame

from keras.layers import MaxPooling1D
from keras.optimizers import Adam
from sklearn import cross_validation
from tensorflow.examples.tutorials.mnist import input_data
import pandas as pd
import sklearn.model_selection
import numpy as np
import keras as K
import matplotlib.pyplot as plt
from keras import regularizers, Model, Input
import random

# from attention import Attention_layer

new_path1 = r'D:\DeepLearning\Keras_Codel\data\indian_结果标签_1.csv'
new_path2 = r'D:\DeepLearning\Keras_Codel\data\indian_label_1.csv'

adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08,  decay=3e-8)


subtrainfeature1 = pd.read_csv(r'D:\DeepLearning\Keras_Codel\data\indian_feature_1.csv')
subtrainLabel1 = pd.read_csv(r'D:\DeepLearning\Keras_Codel\data\indian_label_1_101.csv')
print(subtrainfeature1)

subtrain = pd.merge(subtrainLabel1,subtrainfeature1,on='Id')
from sklearn.utils import shuffle
# subtrain = shuffle(subtrain)
labels = subtrain.Class
subtrain.drop(["Class","Id"], axis=1, inplace=True)
subtrain = subtrain.as_matrix()
(x_train, x_test,y_train,y_test)=cross_validation.train_test_split(subtrain,labels,test_size=0.95,stratify=labels)
# (x_train, x_test,y_train,y_test)=cross_validation.train_test_split(subtrain,labels,test_size=0.5)

print(y_train)
new_path3 = r'D:\DeepLearning\Keras_Codel\data\indian_train2.csv'
# f = open(new_path3,'w')
# csv_write = f.writer(y_train,dialect='excel')
y_train.to_csv(new_path3,index=False,header=False)


# x_train = subtrain[0:10000]
x_test = subtrain[0:]
print(x_test.shape)
# # y_train = labels[0:10000]
y_test = labels[0:]

# x_train = subtrain[0:10000]
# x_test = subtrain[0:]
# # y_train = labels[0:10000]
# y_test = labels[0:]


# y_train = keras.utils.to_categorical(y_train, num_classes=2)
# y_test = keras.utils.to_categorical(y_test, num_classes=2)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

y_train=np_utils.to_categorical(y_train, num_classes=16)
y_test=np_utils.to_categorical(y_test, num_classes=16)
SINGLE_ATTENTION_VECTOR = False
APPLY_ATTENTION_BEFORE_LSTM = False
TIME_STEPS = 5
def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[3])
    a = Permute((3, 1,2))(inputs)
    # a = Reshape((input_dim, TIME_STEPS , TIME_STEPS ))(a) # this line is not useful. It's just to know which dimension is what. softmax
    a = Dense(TIME_STEPS, activation='softmax')(a)
    print(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2,3,1))(a)

    output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    return output_attention_mul


def attention_3d_block_zong(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    print(inputs)
    input_dim = int(inputs.shape[3])
    a = Permute((3, 2,1))(inputs)
    print(a)
    a = Reshape((input_dim, TIME_STEPS, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what. softmax
    a = Dense(TIME_STEPS, activation='softmax')(a)
    print(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 3,1), name='attention_vec')(a)
    output_attention_mul = merge([inputs, a_probs], mode='mul')
    return output_attention_mul


model1_7 = Input( shape=(100,))
# x = Dense(units=256)(model1_7)


x = Reshape((10, 10,1))(model1_7)

x1 = Conv2D(filters=32, kernel_size=3,strides=1,activation='relu', padding='same')(x)
x1 =  BatchNormalization()(x1)
x1 =  Activation('relu')(x1)
x2 = Conv2D(filters=32, kernel_size=3,strides=1,activation='relu', padding='same')(x1)

x3 = concatenate([x1, x2])
x111 = BatchNormalization()(x3)
x111 = Activation('relu')(x111)
x4 = Conv2D(filters=64, kernel_size=3,strides=1,activation='relu', padding='same')(x111)

x5 = concatenate([x3, x4])
x222 = BatchNormalization()(x5)
x222 = Activation('relu')(x222)
x6 = Conv2D(filters=64, kernel_size=3,strides=1,activation='relu', padding='same')(x222)

x7 = concatenate([x5, x6])
x7 = BatchNormalization()(x7)
x7 = Activation('relu')(x7)
x8 = Conv2D(filters=256, kernel_size=3,strides=1,activation='relu', padding='same')(x7)

x8 = BatchNormalization()(x8)
x8 = Activation('relu')(x8)
x9 = Conv2D(filters=256, kernel_size=3,strides=1,activation='relu', padding='same')(x8)
x9 = MaxPooling2D(pool_size=2)(x9)


x_attention1 = attention_3d_block(x9)

# x_attention1 = Reshape((25,124))(x_attention1)
# x_attention1 = Conv2D(filters=124, kernel_size=4,strides=1,activation='relu')(x_attention1)


# x_attention1 = Permute((2,1))(x_attention1)
# x_attention1 = LSTM(40,return_sequences=True)(x_attention1)


x_attention2 = attention_3d_block_zong(x9)

# x_attention2 = Conv2D(filters=124, kernel_size=4,strides=1,activation='relu')(x_attention2)

# x_attention2 = Reshape((25,124))(x_attention2)
# x_attention2 = Permute((2,1))(x_attention2)
# x_attention2 = LSTM(40,return_sequences=True)(x_attention2)

x = concatenate([x_attention1,x_attention2])
# x = Dropout(0.5)(x)


x = Flatten()(x)

x = Dense(256, activation='relu')(x)
all1_output = Dense(16,activation='softmax')(x)
# all1_output = Activation('softmax')(all1_output)
#


model1 = Model(inputs=[model1_7], outputs=[all1_output])
model1.summary()


# from keras.utils.vis_utils import plot_model
# plot_model(model1, to_file='model.png',show_shapes='True')


model1.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# model1.fit(x=x_train,y=y_train,batch_size=500,nb_epoch=200,verbose=2,validation_data=(x_test,y_test))

# loss,acc = model1.evaluate(x_test,y_test,verbose=2)
# print(acc)



from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint(filepath="best_model.h5",#(就是你准备存放最好模型的地方),
                             monitor='val_acc',#(或者换成你想监视的值,比如acc,loss, val_loss,其他值应该也可以,还没有试),
                             verbose=1,#(如果你喜欢进度条,那就选1,如果喜欢清爽的就选0,verbose=冗余的),
                             save_best_only='True',#(只保存最好的模型,也可以都保存),
                             save_weights_only='True',
                             mode='max',#(如果监视器monitor选val_acc, mode就选'max',如果monitor选acc,mode也可以选'max',如果monitor选loss,mode就选'min'),一般情况下选'auto',
                        period=1)#(checkpoints之间间隔的epoch数)
#损失不下降，则自动降低学习率

lrreduce=keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)

import time
fit_start = time.clock()
history= model1.fit(x=x_train,y=y_train,batch_size= 1000,nb_epoch=2,verbose=2,validation_data=(x_test,y_test),callbacks = [checkpoint])
fit_end = time.clock()

print("train time is: ",fit_end-fit_start)

model1.load_weights('best_model.h5')

t_start = time.clock()
loss,acc = model1.evaluate(x_test,y_test,verbose=2)
t_end = time.clock()
print('Test loss :',loss)
print('Test accuracy :',acc)
print("test time is: ",t_end-t_start)





y_pred_class = model1.predict(x_test)

data1 = DataFrame(y_pred_class)
data1.to_csv(r'C:\Users\HASEE\Desktop\3\indian\indian_结果标签.csv',index=False,header=False)


'-----------------------------------------------------------------------------------------------------------------------'

# conv5 = model1.get_layer('conv_5')
# print(conv5)


'----------------------------------------------------------------------------------------------------------------------'
list3 = []


lines = y_pred_class.tolist()



f=open(new_path1, mode='w')

a = 0
for line in lines:
    if line:
        a = a + 1
        # print(line.index(max(line)))
        f.write(str(line.index(max(line))))
        f.write('\n')
        list3.append(line.index(max(line)))
        # list.append('\n')
f.close()

list22 =[ ]
f2 = open(new_path2,mode='r')
list_true = f2.readlines()

for i in list_true:
    ii = i
    ii = ii.replace('\n','')
    ii = int(ii)
    list22.append(ii)

# print(list)

"----------------------------------------------------------------------------------------------------------------------"
all_path = r'C:\Users\HASEE\Desktop\3\indian'

path1 = all_path + '\indian_label.csv'
path2 = all_path + '\indian_结果标签_1.csv'
path3 = all_path + '\indian_label_zui.csv'



f1=open(path1, mode='r')
f2=open(path2, mode='r')
f3=open(path3, mode='w')

line1 = f1.readlines()
line2 = f2.readlines()

a = 0

for i1 in line1:
    if i1 != '16\n':
        i1 = line2[a]
        a = a + 1
        # print(a)
        f3.write(i1)
    else:
        f3.write(i1)
        continue

f3.close()
"----------------------------------------------------------------------------------------------------------------------"
import numpy as np
aa = 0
path3 = all_path + '\indian_label_zui.csv'
f3=open(path3, mode='r')
list = f3.readlines()
for i in range(len(list)):
    aa = aa + 1
    if list[i] == '1\n':
        list[i] =[255,255,102]
    elif list[i] == '2\n':
        list[i] =[0,48,205]
    elif list[i] == '3\n':
        list[i] = [255, 102, 0]
    elif list[i] == '4\n':
        list[i] =[0,255,154]
    elif list[i] == '5\n':
        list[i] =[255,48,205]
    elif list[i] == '6\n':
        list[i] =[102,0,255]
    elif list[i] == '7\n':
        list[i] =[0,154,255]
    elif list[i] == '8\n':
        list[i] =[0,255,0]
    elif list[i] == '9\n':
        list[i] =[129,129,0]
    elif list[i] == '10\n':
        list[i] = [129, 0, 129]
    elif list[i] == '11\n':
        list[i] = [48, 205, 205]
    elif list[i] == '12\n':
        list[i] = [0, 102, 102]
    elif list[i] == '13\n':
        list[i] = [48, 205, 48]
    elif list[i] == '14\n':
        list[i] = [102, 48, 0]
    elif list[i] == '15\n':
        list[i] = [102, 255, 255]
    elif list[i] == '0\n':
        list[i] = [255, 255, 0]
    else:
        list[i] = [0, 0, 0]
data = np.reshape(list, (145, 145, 3))
import scipy.misc
scipy.misc.imsave(r'C:\Users\HASEE\Desktop\3\实验结果\CNN_indian_0.95_AlexNettttt2.jpg', data)

print('11')





from sklearn import metrics


def kappa(confusion_matrix, k):
    dataMat = np.mat(confusion_matrix)
    P0 = 0.0
    for i in range(k):
        P0 += dataMat[i, i]*1.0
    xsum = np.sum(dataMat, axis=1)
    ysum = np.sum(dataMat, axis=0)
    Pe  = float(ysum*xsum)/np.sum(dataMat)**2
    OA = float(P0/np.sum(dataMat)*1.0)
    cohens_coefficient = float((OA-Pe)/(1-Pe))
    return cohens_coefficient

classify_report = metrics.classification_report(list22, list3)
confusion_matrix = metrics.confusion_matrix(list22, list3)
overall_accuracy = metrics.accuracy_score(list22, list3)
acc_for_each_class = metrics.precision_score(list22, list3, average=None)
average_accuracy = np.mean(acc_for_each_class)
kappa_coefficient = kappa(confusion_matrix, 16)
print('classify_report : \n', classify_report)
print('confusion_matrix : \n', confusion_matrix)
print('acc_for_each_class : \n', acc_for_each_class)
print('average_accuracy: {0:f}'.format(average_accuracy))
print('overall_accuracy: {0:f}'.format(overall_accuracy))
print('kappa coefficient: {0:f}'.format(kappa_coefficient))

newpath = r'C:\Users\HASEE\Desktop\3\实验结果\CNN_indian_0.95_AlexNetttt2.txt'
f = open(newpath,'w')
f.write(classify_report)
# f.write(confusion_matrix)
f.write(str(acc_for_each_class.tolist()))
f.write('\n')
f.write('average_accuracy:{0:f}'.format(average_accuracy))
f.write('\n')
f.write('overall_accuracy:{0:f}'.format(overall_accuracy))
f.write('\n')
f.write('kappa coefficient:{0:f}'.format(kappa_coefficient))
f.write('\n')

f.close()