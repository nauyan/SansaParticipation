import glob
#import cv2
import pandas as pd 
import numpy as np
import shutil  
import os
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from sklearn.model_selection import train_test_split
import keras

from zipfile import ZipFile
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

import efficientnet.keras as efn 
#from keras.applications.resnext import ResNeXt50
from keras.applications.resnet import ResNet50
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.densenet import DenseNet121
from keras.applications.inception_resnet_v2 import InceptionResNetV2

from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D

import shutil
import json

im_width = 384
im_height = 384
epochs = 50
dropout = 0.1
batch_size = 16

dataset_dir = "dataset/train_set/"
"""

images = glob.glob(dataset_dir+"*/*.jpeg")
#print(len(images))
#print(os.path.basename(os.path.dirname(images[0])))
Images = []
Labels = []
for path in images:
    img = img_to_array(load_img(path, color_mode='rgb'))
    img = img/255.0
    
    if os.path.basename(os.path.dirname(path))=="positive":
        label = 1
    else:
        label = 0
    #dataset.append([img,label])
    Images.append(img)
    Labels.append(label)

Images = np.array(Images)
print(Images.shape)

Labels = np.array(Labels)
print(Labels.shape)
#print(np.unique(Labels, return_counts=True))

# Test and Train Split of Data
X_train, X_test, y_train, y_test = train_test_split(Images, Labels, test_size=0.20)
print(X_train.shape,y_train.shape)
print(X_test.shape,y_test.shape)
"""
datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2)

train_generator = datagen.flow_from_directory(
        dataset_dir, target_size=(im_height,im_width),
        color_mode="grayscale",
        subset='training', class_mode='binary', batch_size=batch_size)

val_generator = datagen.flow_from_directory(
        dataset_dir, target_size=(im_height,im_width),
        color_mode="grayscale",
        subset='validation', class_mode='binary', batch_size=batch_size)

steps_per_epoch = 400
validation_steps = 100



# Training Model Layers Arrangment
#baseModel = efn.EfficientNetB0(weights=None, include_top=False,input_shape=(im_width,im_height,1))
#baseModel = MobileNetV2(weights=None, include_top=False,input_shape=(im_width,im_height,3))
#baseModel = DenseNet121(weights=None, include_top=False,input_shape=(im_width,im_height,3))
baseModel = InceptionResNetV2(weights=None, include_top=False,input_shape=(im_width,im_height,1))

headModel = baseModel.output
headModel = GlobalAveragePooling2D(name='avg_pool')(headModel)
headModel = Dropout(dropout, name='top_dropout')(headModel)
headModel = Dense(1,activation='sigmoid')(headModel)
model = Model(inputs=baseModel.input, outputs=headModel)

# Loss Function and its Parameters amsgrad=True
adam = keras.optimizers.Adam(amsgrad=True)
sgd = keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=True)

# Compilation of Model
model.compile(loss='binary_crossentropy',optimizer=adam,metrics=['accuracy'])

# Setting up of Callbacks for the Model
callbacks = [
    EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1),
    ModelCheckpoint('./Best.h5', monitor='val_loss', mode = 'min' , verbose=1, save_best_only=True, save_weights_only=False)
]

# Seting up of Data Augmentation
#aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
	#width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
	#fill_mode="nearest")

# Setting up of Training Parameters and Starting Training    
#results = model.fit_generator(aug.flow(X_train, y_train, batch_size=batch_size),
	#validation_data=(X_test, y_test), epochs=epochs, callbacks=callbacks)
#results = model.fit(X_train, y_train, batch_size=batch_size,validation_data=(X_test, y_test), epochs=epochs, callbacks=callbacks)
results = model.fit_generator(
        train_generator, 
        steps_per_epoch = steps_per_epoch, validation_steps = validation_steps,
        epochs=epochs,
        validation_data=val_generator, 
        callbacks=callbacks,
        use_multiprocessing=True)

model.save('./Last_Epoch.h5') 
# Evaluting Model 
#print(model.evaluate(X_test, y_test, verbose=1))

# Train and Test Validation Loss Plots
plt.figure(figsize=(8, 8))
plt.title("Learning curve")
plt.plot(results.history["loss"], label="loss")
plt.plot(results.history["val_loss"], label="val_loss")
plt.plot( np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
plt.xlabel("Epochs")
plt.ylabel("log_loss")
plt.legend();
plt.savefig('./train_loss.png')

# Train and Test Accuracy Loss Plots
plt.figure(figsize=(8, 8))
plt.title("Learning curve")
plt.plot(results.history["accuracy"], label="accuracy")
plt.plot(results.history["val_accuracy"], label="val_accuracy")
plt.plot( np.argmax(results.history["val_accuracy"]), np.max(results.history["val_accuracy"]), marker="x", color="r", label="best model")
plt.xlabel("Epochs")
plt.ylabel("accuracy")
plt.legend();
plt.savefig('./train_accuracy.png')
""""""