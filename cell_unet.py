# -*- coding: utf-8 -*-


import tensorflow as tf
img_width, img_height,color_channels = 128,128,3
from keras.api._v2.keras.utils import Sequence
import glob

#convert pixel to float (div by 255)

#build model


import os 
import random
import numpy as np
from tqdm import tqdm 
from skimage.io import imread,imshow
from skimage.transform import resize 
import matplotlib.pyplot as plt 
import cv2
import glob
from tensorflow.python import keras

# train_path = 'bdd100k_images_10k/bdd100k/images/10k/train/'
# test_path = 'bdd100k_images_10k/bdd100k/images/10k/test/'
# train_ids = next(os.walk(train_path))
# test_ids = next(os.walk(test_path))
# train_ids_rm = []
# test_ids_rm = []

# for item in train_ids[2]:
#   item = item.replace(".jpg","")
#   train_ids_rm.append(item)
# for item in test_ids[2]:
#   item = item.replace(".jpg","")
#   test_ids_rm.append(item)

img_width,img_height,img_channels = 128,128,3
# X_train = np.zeros((len(train_ids_rm),img_height,img_width,img_channels),dtype=np.uint8)
# y_train = np.zeros((len(train_ids_rm),img_height,img_width,1),dtype=bool)
# X_test = np.zeros((len(train_ids_rm),img_height,img_width,img_channels),dtype=np.uint8)

# mask_path = next(os.walk("bdd100k_drivable_labels_trainval/colormaps/train/"))[2]
# mask_path_rm = []
# for item in mask_path:
#   item = item.replace(".png","")
#   mask_path_rm.append(item)

print('resizing images and masks....')
image_paths = os.listdir('./bdd100k_images_10k/bdd100k/images/10k/train')
image_paths += os.listdir('./bdd100k_images_10k/bdd100k/images/10k/test')
image_paths += os.listdir('./bdd100k_images_10k/bdd100k/images/10k/val')
image_paths = [i.split('.')[0] for i in image_paths]

train_labels = os.listdir('bdd100k_drivable_labels_trainval/colormaps/train')
train_labels += os.listdir('bdd100k_drivable_labels_trainval/colormaps/train')
train_labels = [i.split('.')[0] for i in train_labels]

count = 0
# for i in image_paths:
#     if i not in train_labels:
#         count += 1
#     else:
#         print(i)
#         break

# print("count = ", count)
train_paths = glob.glob(f"{os.path.join('./bdd100k_images_10k/bdd100k/images/10k', 'train')}/*.jpg")
train_paths += glob.glob(f"{os.path.join('./bdd100k_images_10k/bdd100k/images/10k', 'test')}/*.jpg")
train_paths += glob.glob(f"{os.path.join('./bdd100k_images_10k/bdd100k/images/10k', 'val')}/*.jpg")
train_paths = [i.split('.jpg')[0] for i in train_paths]
ids = [train_paths[i].split('\\')[-1] for i in range(len(train_paths))]
test_paths = glob.glob(f"{os.path.join('bdd100k_drivable_labels_trainval/colormaps', 'train')}/*.png")
test_paths += glob.glob(f"{os.path.join('bdd100k_drivable_labels_trainval/colormaps', 'val')}/*.png")
test_paths = [i.split('.png')[0] for i in test_paths]
valid_ids = [i for i in [*set([i.split('\\')[-1] for i in test_paths])] if i in ids]
test_paths = [i for i in test_paths if i.split('\\')[-1] in valid_ids]
# print(len(test_paths))
train_paths = [i for i in train_paths if i.split('\\')[-1] in valid_ids]
# print(len(train_paths))


# core = cv2.imread('./bdd100k_images_10k/bdd100k/images/10k/train/00054602-3bf57337.jpg')
# img = cv2.imread('bdd100k_drivable_labels_trainval/colormaps/train/00054602-3bf57337.png')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# plt.figure(figsize=(12, 6))
# plt.subplot(121)
# plt.imshow(img, cmap='gray')
# plt.subplot(122)
# plt.imshow(core)
# plt.show()

def keyFunc(x):
    return x.split('\\')[-1]
train_paths.sort(key = keyFunc)
test_paths.sort(key = keyFunc)
count = 0
for idx, i in enumerate(train_paths):
    if test_paths[idx].split('\\')[-1] != i.split('\\')[-1]:
        count += 1
# print(count)

class LoadData(Sequence):
    def __init__(self, trainPaths, testPaths, batchSize):       
        self.trainPaths = trainPaths
        self.testPaths = testPaths
        self.batchSize = batchSize
    
    def __len__(self):
        return int(np.ceil(len(self.trainPaths)/self.batchSize))
    
    def readImage(self, path):
        '''
        INPUTS:
            path: the file path to the image that we need to read
            
        OUTPUTS:
            we read the image in the given path and normalize it by dividing it by 255.0
        '''
        return cv2.imread(path)/255.0
    
    def grayScale(self, image):
        image *= 255.0 
        grayValue = 0.07 * image[:,:,2] + 0.72 * image[:,:,1] + 0.21 * image[:,:,0]
        gray_img = grayValue.astype(np.uint8)
        return gray_img
    
    def __getitem__(self, idx):
        x_paths = self.trainPaths[idx * self.batchSize : (idx + 1) * self.batchSize]
        y_paths = self.testPaths[idx * self.batchSize : (idx + 1) *self.batchSize]
        
        x_paths = [i + '.jpg' for i in x_paths]
        y_paths = [i + '.png' for i in y_paths]
        
        print(y_paths[0])
        
        x = np.array(list(map(self.readImage, x_paths)))
        y = np.array(list(map(self.readImage, y_paths)))
        y = np.array(list(map(self.grayScale, y)))
               
        return x, y
testing = LoadData(train_paths, test_paths, 4).__iter__()
for i, j in testing:
    # i = np.squeeze(i,0)
    # j = np.squeeze(j,0)
    # print(i.shape)
    # print(j.shape)
    i = resize(i,(4,img_height,img_width),mode='constant',preserve_range=True)
    j = resize(j,(4,img_height,img_width),mode="constant",preserve_range=True)
    j = np.expand_dims(j,axis=-1)
    break

print(i[0].shape)
print(j[0].shape)
# plt.imshow(j[0],cmap="gray")


print("finished resizing....")



X_train = np.zeros((len(train_paths),img_height,img_width,img_channels),dtype=np.uint8)
y_train = np.zeros((len(train_paths),img_height,img_width,1),dtype=np.bool)

#constrictor
inputs = tf.keras.layers.Input((img_width,img_height,color_channels))
s = tf.keras.layers.Lambda(lambda x:x/255)(inputs)
c1 = tf.keras.layers.Conv2D(16,(3,3),activation="relu",kernel_initializer="he_normal",padding="same")(s)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16,(3,3),activation="relu",kernel_initializer="he_normal",padding="same")(c1)
p1 = tf.keras.layers.MaxPooling2D((2,2))(c1)

c2 = tf.keras.layers.Conv2D(32,(3,3),activation="relu",kernel_initializer="he_normal",padding="same")(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32,(3,3),activation="relu",kernel_initializer="he_normal",padding="same")(c2)
p2 = tf.keras.layers.MaxPooling2D((2,2))(c2)

c3 = tf.keras.layers.Conv2D(64,(3,3),activation="relu",kernel_initializer="he_normal",padding="same")(p2)
c3 = tf.keras.layers.Dropout(0.1)(c3)
c3 = tf.keras.layers.Conv2D(64,(3,3),activation="relu",kernel_initializer="he_normal",padding="same")(c3)
p3 = tf.keras.layers.MaxPooling2D((2,2))(c3)

c4 = tf.keras.layers.Conv2D(128,(3,3),activation="relu",kernel_initializer="he_normal",padding="same")(p3)
c4 = tf.keras.layers.Dropout(0.1)(c4)
c4 = tf.keras.layers.Conv2D(128,(3,3),activation="relu",kernel_initializer="he_normal",padding="same")(c4)
p4 = tf.keras.layers.MaxPooling2D((2,2))(c4)

c5 = tf.keras.layers.Conv2D(256,(3,3),activation="relu",padding="same",kernel_initializer="he_normal")(p4)
c5 = tf.keras.layers.Dropout(0.1)(c5)
c5 = tf.keras.layers.Conv2D(256,(3,3),activation="relu",padding="same",kernel_initializer="he_normal")(c5)

#upsampling 
u6 = tf.keras.layers.Conv2DTranspose(128,(2,2),strides=(2,2),padding="same")(c5)
u6 = tf.keras.layers.concatenate([u6,c4])
c6 = tf.keras.layers.Conv2D(128,(3,3),activation="relu",kernel_initializer="he_normal",padding="same")(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128,(3,3),activation="relu",kernel_initializer="he_normal",padding="same")(c6)

u6 = tf.keras.layers.Conv2DTranspose(128,(2,2),strides=(2,2),padding="same")(c5)
u6 = tf.keras.layers.concatenate([u6,c4])
c6 = tf.keras.layers.Conv2D(128,(3,3),activation="relu",kernel_initializer="he_normal",padding="same")(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128,(3,3),activation="relu",kernel_initializer="he_normal",padding="same")(c6)

u7 = tf.keras.layers.Conv2DTranspose(128,(2,2),strides=(2,2),padding="same")(c6)
u7 = tf.keras.layers.concatenate([u7,c3])
c7 = tf.keras.layers.Conv2D(128,(3,3),activation="relu",kernel_initializer="he_normal",padding="same")(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(128,(3,3),activation="relu",kernel_initializer="he_normal",padding="same")(c7)

u8 = tf.keras.layers.Conv2DTranspose(128,(2,2),strides=(2,2),padding="same")(c7)
u8 = tf.keras.layers.concatenate([u8,c2])
c8 = tf.keras.layers.Conv2D(128,(3,3),activation="relu",kernel_initializer="he_normal",padding="same")(u8)
c8 = tf.keras.layers.Dropout(0.2)(c8)
c8 = tf.keras.layers.Conv2D(128,(3,3),activation="relu",kernel_initializer="he_normal",padding="same")(c8)

u9 = tf.keras.layers.Conv2DTranspose(128,(2,2),strides=(2,2),padding="same")(c8)
u9 = tf.keras.layers.concatenate([u9,c1])
c9 = tf.keras.layers.Conv2D(128,(3,3),activation="relu",kernel_initializer="he_normal",padding="same")(u9)
c9 = tf.keras.layers.Dropout(0.2)(c9)
c9 = tf.keras.layers.Conv2D(128,(3,3),activation="relu",kernel_initializer="he_normal",padding="same")(c9)

outputs = tf.keras.layers.Conv2D(1,(1,1),activation="sigmoid")(c9)

model = tf.keras.Model(inputs=[inputs],outputs=[outputs])
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])
model.summary()

#######################################################################
#model checkpoint
checkpointer = tf.keras.callbacks.ModelCheckpoint('for_nuclei.h5',verbose=1,save_best_only=True)
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2,monitor="val_loss"),
    tf.keras.callbacks.TensorBoard(log_dir='logs')
]

results = model.fit(X_train,y_train,validation_split=0.1,batch_size = 16,epochs = 50,callbacks=callbacks)

# idx = random.randint(0,len(X_train))
# preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)],verbose=1)
# preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):],verbose=1)
# preds_test = model.predict(X_test,verbose = 1)
# seed = 40
# random.seed = seed

# preds_train_t = (preds_train > 0.5).astype(np.uint8)
# preds_val_t = (preds_val > 0.5).astype(np.uint8)
# preds_test_t = (preds_test > 0.5).astype(np.uint8)


# ix = random.randint(0,len(preds_train_t))
# imshow(X_train[ix])
# plt.show()
# imshow(np.squeeze(y_train[ix].astype(float)))
# plt.show()
# imshow(np.squeeze(preds_train_t[ix]))
# plt.show()


# ix = random.randint(0,len(preds_val_t))
# imshow(X_train[int(X_train.shape[0]*0.9):][ix])
# plt.show()
# imshow(np.squeeze(y_train[int(y_train.shape[0]*0.9):][ix].astype(float)))
# plt.show()
# imshow(np.squeeze(preds_val_t[ix]))
# plt.show()

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard

# Commented out IPython magic to ensure Python compatibility.
# %tensorboard --logdir "/content/logs" --port 8088
# TO VIEW THE VAL_LOSS AND ACCURACY GRPAHS, RUN TENSORBOARD ON A PORT ON LOCALHOST


