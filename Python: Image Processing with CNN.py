# -*- coding: utf-8 -*-

# %reset -f

import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

!wget --no-check-certificate \
https://storage.googleapis.com/laurencemoroney-blog.appspot.com/happy-or-sad.zip \
-O /tmp/happy_or_sad.zip

import os
import zipfile

local_zip = '/tmp/happy_or_sad.zip'

zip_ref = zipfile.ZipFile(local_zip, 'r')

zip_ref.extractall('/tmp')
zip_ref.close()

# shutil.rmtree('/tmp/happy_and_sad/train/happy')
# shutil.rmtree('/tmp/happy_and_sad/train/sad')
# shutil.rmtree('/tmp/happy_and_sad/validation/happy')
# shutil.rmtree('/tmp/happy_and_sad/validation/sad')

os.makedirs('/tmp/happy_and_sad')
os.makedirs('/tmp/happy_and_sad/train')
os.makedirs('/tmp/happy_and_sad/validation')
os.makedirs('/tmp/happy_and_sad/train/happy')
os.makedirs('/tmp/happy_and_sad/train/sad')
os.makedirs('/tmp/happy_and_sad/validation/happy')
os.makedirs('/tmp/happy_and_sad/validation/sad')

happy_dir = '/tmp/happy'
sad_dir = '/tmp/sad'

happy_fnames = os.listdir('/tmp/happy')
sad_fnames = os.listdir('/tmp/sad')

import random

random.seed(42)
happy_train_idx = random.sample(happy_fnames, 32)
sad_train_idx = random.sample(sad_fnames, 32)
happy_test_idx = list(set(happy_fnames) - set(happy_train_idx))
sad_test_idx = list(set(sad_fnames) - set(sad_train_idx))

import shutil

for file_name in happy_fnames:
    full_file_name = os.path.join(happy_dir, file_name)
    if os.path.isfile(full_file_name):
      shutil.copy(full_file_name, '/tmp/happy_and_sad/train/happy')
        
for file_name in happy_test_idx:
    full_file_name = os.path.join(happy_dir, file_name)
    if os.path.isfile(full_file_name):
      shutil.copy(full_file_name, '/tmp/happy_and_sad/validation/happy')
  
for file_name in sad_fnames:
    full_file_name = os.path.join(sad_dir, file_name)
    if os.path.isfile(full_file_name):
      shutil.copy(full_file_name, '/tmp/happy_and_sad/train/sad')
        
for file_name in sad_test_idx:
    full_file_name = os.path.join(sad_dir, file_name)
    if os.path.isfile(full_file_name):
      shutil.copy(full_file_name, '/tmp/happy_and_sad/validation/sad')

base_dir = '/tmp/happy_and_sad'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# Directory with our training cat/dog pictures
train_happy_dir = os.path.join(train_dir, 'happy')
train_sad_dir = os.path.join(train_dir, 'sad')

# Directory with our validation cat/dog pictures
validation_happy_dir = os.path.join(validation_dir, 'happy')
validation_sad_dir = os.path.join(validation_dir, 'sad')

train_happy_fnames = os.listdir( train_happy_dir )
train_sad_fnames = os.listdir( train_sad_dir )

print(train_happy_fnames[:10])
print(train_sad_fnames[:10])

print('total training happy images :', len(os.listdir(train_happy_dir ) ))
print('total training sad images :', len(os.listdir(train_sad_dir ) ))

print('total validation happy images :', len(os.listdir( validation_happy_dir ) ))
print('total validation sad images :', len(os.listdir( validation_sad_dir ) ))

happy_pix = [os.path.join(happy_dir, fname) 
                for fname in happy_fnames 
               ]

for i, img_path in enumerate(happy_pix):
  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()

import tensorflow as tf

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2), 
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results into a one dimension data to feed into a DNN
    tf.keras.layers.Flatten(), 
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('cats') and 1 for the other ('dogs')
    # Note that because we are facing a two-class classification problem, i.e. a binary classification problem, we will 
    # end our network with a sigmoid activation, so that the output of our network will be a single scalar between 0 and 1,
    # encoding the probability that the current image is class 1 (as opposed to class 0).
    tf.keras.layers.Dense(1, activation='sigmoid')  
])

from tensorflow.keras.optimizers import RMSprop

model.compile(optimizer=RMSprop(lr=0.001), #learning rate of 0.001. In this case, using the RMSprop optimization algorithm is preferable to stochastic gradient descent (SGD), because RMSprop automates learning-rate tuning for us.
              loss='binary_crossentropy', #binary_crossentropy loss, because it's a binary classification problem and our final activation is a sigmoid.
              metrics = ['acc'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255.
# Flow training images in batches of 20 using generator

train_datagen = ImageDataGenerator(rescale=1./255,
      # rotation_range=40, #These 7 rows are augmentations to increase our dataset by twisting the images
      # width_shift_range=0.2, 
      # height_shift_range=0.2,
      # shear_range=0.2,
      # zoom_range=0.2,
      # horizontal_flip=True,
      # fill_mode='nearest'
      )

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=20,
                                                    class_mode='binary',
                                                    target_size=(300, 300))


validation_datagen = ImageDataGenerator(rescale=1./255,
      # rotation_range=40, #These 7 rows are augmentations to increase our dataset by twisting the images
      # width_shift_range=0.2,
      # height_shift_range=0.2,
      # shear_range=0.2,
      # zoom_range=0.2,
      # horizontal_flip=True,
      # fill_mode='nearest'
      )

validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                              batch_size=20,
                                                              class_mode='binary',
                                                              target_size=(300, 300))

history = model.fit_generator(train_generator,
                              validation_data=validation_generator,
                              epochs=10,
                              verbose=1)

#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc      = history.history[     'acc' ]
val_acc  = history.history[ 'val_acc' ]
loss     = history.history[    'loss' ]
val_loss = history.history['val_loss' ]

epochs   = range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot  ( epochs,     acc )
plt.plot  ( epochs, val_acc )
plt.title ('Training and validation accuracy')
plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot  ( epochs,     loss )
plt.plot  ( epochs, val_loss )
plt.title ('Training and validation loss'   )

from google.colab import files
from keras.preprocessing import image

uploaded=files.upload()

predict_result = []

for fn in uploaded.keys():
 
  # predicting images
  path='/content/' + fn
  img=image.load_img(path, target_size=(300, 300))
  
  x=image.img_to_array(img)
  x=np.expand_dims(x, axis=0)
  images = np.vstack([x])
  
  classes = model.predict(images, batch_size=2)
  
  print(classes[0])
  predict_result.append(classes[0])
  if classes[0]>0:
    print(fn + " is a happy face")
    
  else:
    print(fn + " is a sad face")

from sklearn.metrics import accuracy_score

true_result = ["sad", "happy", "sad", "sad", "happy", "happy", "happy", "happy", "sad", "happy"]

for i in range(0,10):
  if true_result[i] == "sad":
    true_result[i] = 0
  else:
    true_result[i] = 1

accuracy_score(true_result, predict_result)

# With Callback
ACCURACY_THRESHOLD = 0.995

class myCallback(tf.keras.callbacks.Callback): 
    def on_epoch_end(self, epoch, logs={}): 
      if(logs.get('acc') > ACCURACY_THRESHOLD):
        print("\nReached %2.2f%% accuracy, so stopping training!!" %(ACCURACY_THRESHOLD*100))   
        self.model.stop_training = True

callbacks = myCallback()

history_new = model.fit_generator(train_generator,
                              validation_data=validation_generator,
                              epochs=10,
                              verbose=1,
                              callbacks = [callbacks])

