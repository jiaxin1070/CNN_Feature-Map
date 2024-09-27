# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 13:37:38 2022

@author: user
"""

#import東東
import cv2
import numpy as np
import random
import tensorflow as tf 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.datasets import load_digits

#讀取資料集
digits = load_digits()
y=digits.target

#將digits的圖片存到imgs
imgs=[]
for i in range(len(digits.images)):
    imgs.append(digits.images[i])

#印出imgs的前100張照片
plt.rcParams["figure.figsize"] = (18,18)
plt.gray() 
for i in range(100):
    plt.subplot(20, 20, i + 1)
    plt.imshow(imgs[i], cmap=plt.cm.gray, vmax=16, interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
plt.show() 

#中值濾波
med=[]
for i in range(len(imgs)):
    med.append(cv2.medianBlur(imgs[i],1))
med=np.array(med)

#形態學
kernel = np.ones((2,2), np.uint8)
ero=[]
dil=[]
mix=[]
for i in range(len(med)):
    #侵蝕
    erosion=cv2.erode(med[i],kernel,iterations = 1)
    ero.append(erosion)
    #膨脹
    dil.append(cv2.dilate(med[i],kernel,iterations = 1))
    #先侵蝕再膨脹
    mix.append(cv2.dilate(erosion,kernel,iterations = 1))

#list轉成array
ero=np.array(ero)
dil=np.array(dil)
mix=np.array(mix)

#做gradient
grad1=mix-ero 
grad2=ero-mix

#將以上前處理的資料合併
pic=np.r_[med,ero,dil,mix,grad1,grad2]
lab=np.r_[y,y,y,y,y,y]

#將它們合併在一個列表中
comb_list = list(zip(pic,lab))
random.seed(45)
#均等地洗牌
random.shuffle(comb_list)
#再次分開他們
images_features, images_targets = zip(*comb_list)

#分資料
X_train, X_test_val, y_train, y_test_val = train_test_split(
    images_features, images_targets, test_size = 0.5, random_state = 45)

#X資料標準化
X_train = tf.keras.utils.normalize(np.asfarray(X_train))
X_test_val = tf.keras.utils.normalize(np.asfarray(X_test_val))

#加入培訓和測試，以相同的方式對所有類別進行編碼（get_dummies） 
Y = y_train + y_test_val
df_y = pd.DataFrame(Y) 
#重新轉為數組
yy= np.asfarray(df_y)
#編輯為獨熱編碼
onehotencoder = OneHotEncoder(categories='auto')
df_y_hot = onehotencoder.fit_transform(yy).toarray()
#再次分成y_train、y_test
y_train = np.asfarray( df_y_hot[:len(y_train)] )
y_test_val= np.asfarray( df_y_hot[len(y_train):] )

#拆分測試集、確認集
X_test, X_val, y_test, y_val = train_test_split(
    X_test_val, y_test_val, test_size = 0.2, random_state = 45)

#將x資料reshape
x_train=np.reshape(X_train,[5391,8,8,1])
x_test=np.reshape(X_test,[4312,8,8,1])
x_val=np.reshape(X_val,[1079,8,8,1])

#建構CNN結構
Con1=Conv2D(128,(2,2),strides=1,padding='same',
            activation = 'relu', input_shape = x_train.shape[1:])
pool1=MaxPooling2D(pool_size = (1,1),strides=1)
Con2=Conv2D(64,(2,2),strides=1,padding='same', activation = 'relu')
pool2=MaxPooling2D(pool_size = (1,1),strides=1)
flat=Flatten()
dense=Dense(16, activation='relu')
logits=Dense(10, activation='softmax')

model = Sequential()
#第一層
model.add(Con1)
model.add(pool1)
model.add(Dropout(0.25))
#第二層
model.add(Con2)
model.add(pool2)
model.add(Dropout(0.25))
#攤平
model.add(flat)
model.add(dense)
model.add(logits)
model.compile(optimizer="adam",loss='categorical_crossentropy',metrics=['accuracy'])
hist=model.fit(x_train, y_train, epochs=50, batch_size=120,
               validation_data=(x_val, y_val))

#畫準確率的圖
plt.figure(figsize=(22,20))
plt.plot(hist.history['accuracy'], 'b', label='train')
plt.plot(hist.history['val_accuracy'], 'r', label='valid')
plt.legend()
plt.grid(True)
plt.title('accuracy')
plt.show()

#feature map

#將model每層的資料存下來
successive_outputs = [layer.output for layer in model.layers[1:]]
#建構視覺化模型
visualization_model = tf.keras.models.Model(
    inputs = model.input, outputs = successive_outputs)

#導入要測試的照片
img=imgs[100]
plt.imshow(imgs[100], "gray")
plt.show

#將測試資料做些處理
x=img_to_array(img)   
x= x.reshape((1,) + x.shape)
x/= 255.0

#將導入圖片拿去做預測
successive_feature_maps = visualization_model.predict(x)
#將CNN結構每層的名字存下來
layer_names = [layer.name for layer in model.layers]

#將預測好的結果存下來
f=[]
for i in range(len(successive_feature_maps)):
    f.append(successive_feature_maps[i])
    
#選取第一層、第二層的Convolution layer的名字
n=[layer_names[0],layer_names[3]]
ff=[f[0],f[3]]

#印出feature map
for i in range(len(n)):
    n_features = ff[i].shape[-1]
    size       = ff[i].shape[ 1]
    display_grid = np.zeros((size, size * n_features))
    for j in range(n_features):
        x  = ff[i][0, :, :, j]
        x -= x.mean()
        x /= x.std ()
        x *=  64
        x += 128
        x  = np.clip(x, 0, 255).astype('uint8')
        # Tile each filter into a horizontal grid
        display_grid[:, j * size : (j + 1) * size] = x
    #Display the grid
    scale = 20. / n_features
    plt.figure( figsize=(scale * n_features, scale) )
    plt.title ( n[i])
    plt.grid  ( False )
    plt.imshow( display_grid, aspect='auto', cmap='viridis' )


