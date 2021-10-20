# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 14:50:21 2021

@author: HZU
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import pandas as pd
import time
import numpy as np
import matplotlib.image as img
from scipy import signal
from PIL import Image
from sklearn.model_selection import train_test_split
import time


train_folder = os.path.abspath('./train_128')
# Flow training images in batches of 32 using train_datagen generator
train_generator = tf.keras.utils.image_dataset_from_directory(
                    directory = train_folder,
                    image_size = (128,128),
                    # image_size = (256,256),
                    color_mode = 'grayscale',                    
                    batch_size = 32,
                    shuffle = True,
                    seed = 42
                    )
    
validation_folder = os.path.abspath('./val_128')
# Flow training images in batches of 32 using train_datagen generator
validation_generator = tf.keras.utils.image_dataset_from_directory(
                    directory = validation_folder,
                    image_size = (128,128),
                    # image_size = (256,256),                    
                    color_mode = 'grayscale',
                    batch_size = 32,
                    shuffle = True,
                    seed = 42
                    )


AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_generator.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = validation_generator.cache().prefetch(buffer_size=AUTOTUNE)


num_classes = 2

model = tf.keras.Sequential([
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(128, 128, 1)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

history = model.fit(train_ds, validation_data=val_ds, epochs=50, batch_size=32)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(50)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()





# ##This part save the model
# model.save('cnn_model.h5')

# ##This part load the model
# model = tf.keras.models.load_model('cnn_model.h5')

import cv2


test_no_folder_path = os.path.abspath('./test_512/no_signature')
test_yes_folder_path = os.path.abspath('./test_512/signature')


def load_names(folder_path):
    all_files = os.listdir(folder_path)
    file_id = []
    for i, name in enumerate(all_files):
        test_file_name = name[:-4]
        file_id.append(test_file_name)
    return file_id

test_no_id = load_names(test_no_folder_path)
test_yes_id = load_names(test_yes_folder_path)

test_no_data = []
for i, name in enumerate(test_no_id):
    file_path = os.path.join(test_no_folder_path, name+'.jpg')
    test_no_data.append(cv2.imread(file_path, 0))

test_yes_data = []
for i, name in enumerate(test_yes_id):
    file_path = os.path.join(test_yes_folder_path, name+'.jpg')
    test_yes_data.append(cv2.imread(file_path, 0))



prediction = model.predict(test_no_data[0][None,:,:])



####################
#Loading datos de challenge


challenge_folder_path = os.path.abspath('./challenge/128')

def load_names(folder_path):
    all_files = os.listdir(folder_path)
    file_id = []
    for i, name in enumerate(all_files):
        test_file_name = name[:-4]
        file_id.append(test_file_name)
    return file_id

challenge_id = load_names(challenge_folder_path)

challenge_data = []
for i, name in enumerate(challenge_id):
    file_path = os.path.join(challenge_folder_path, name+'.jpg')
    challenge_data.append(cv2.imread(file_path, 0))

## Testing the model
import pandas as pd

prediction = []
for i in range(len(challenge_data)):
    value = model.predict(challenge_data[i][None,:,:])
    prediction.append(value)

results_test = []
for i in range(len(prediction)):
    if prediction[i][0][0] > prediction[i][0][1]:
        signature = 0
    else:
        signature = 1
    results_test.append(signature)
    
    
resutls_model = pd.DataFrame(list(zip(challenge_id, results_test)), columns =['Id', 'Expected'])    
resutls_model.to_csv('resutls_model_v4_p_Harold.csv', index=False)









