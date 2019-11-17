# This code block has all the required inputs
import os
import zipfile
import random
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copy

!wget --no-check-certificate \
    "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip" \
    -O "/tmp/cats-and-dogs.zip"

local_zip = '/tmp/cats-and-dogs.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp')
zip_ref.close()

#directory for cats-v-dogs, and subdirectories for training
# and testing. These in turn will need subdirectories for 'cats' and 'dogs'
try:
    os.makedirs('/tmp/cats-v-dogs/testing/cats/')
    os.makedirs('/tmp/cats-v-dogs/testing/dogs/')
    os.makedirs('/tmp/cats-v-dogs/training/cats/')
    os.makedirs('/tmp/cats-v-dogs/training/dogs/')

except OSError:
    pass

def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
  
  src_files = os.listdir(SOURCE)
  
  path, dirs, files = next(os.walk(SOURCE))
  random.shuffle(files)
  file_count = len(files)
  trainFile = float(file_count)*0.9;
  for file_name in range(0, round(trainFile)):
    full_file_name = os.path.join(SOURCE, src_files[file_name])
    size = os.path.getsize(full_file_name)
    if size > 0:
      copy(full_file_name, TRAINING)
    else:
      print (full_file_name +" is zero length, so ignoring")
  for file_name in range(round(trainFile), file_count):
    full_file_name = os.path.join(SOURCE, src_files[file_name])
    size = os.path.getsize(full_file_name)
    if size > 0:
      copy(full_file_name, TESTING)
    else:
      print (full_file_name +" is zero length, so ignoring")

        
# YOUR CODE STARTS HERE for file_name in src_files:
  
# YOUR CODE ENDS HERE


CAT_SOURCE_DIR = "/tmp/PetImages/Cat/"
TRAINING_CATS_DIR = "/tmp/cats-v-dogs/training/cats/"
TESTING_CATS_DIR = "/tmp/cats-v-dogs/testing/cats/"
DOG_SOURCE_DIR = "/tmp/PetImages/Dog/"
TRAINING_DOGS_DIR = "/tmp/cats-v-dogs/training/dogs/"
TESTING_DOGS_DIR = "/tmp/cats-v-dogs/testing/dogs/"

split_size = .9
split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)


model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(), 
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'), 
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('cats') and 1 for the other ('dogs')
    tf.keras.layers.Dense(1, activation='sigmoid')  
])

model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>0.95):
      print("\nReached 95% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()
TRAINING_DIR = "/tmp/cats-v-dogs/training/"


train_datagen = ImageDataGenerator( rescale = 1.0/255,
                                   rotation_range=45,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
test_datagen  = ImageDataGenerator( rescale = 1.0/255. )

VALIDATION_DIR = "/tmp/cats-v-dogs/testing/"

train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                    batch_size=30,
                                                    class_mode='binary',
                                                    target_size=(150, 150))     

validation_generator =  test_datagen.flow_from_directory(VALIDATION_DIR,
                                                         batch_size=30,
                                                         class_mode  = 'binary',
                                                         target_size = (150, 150))


history = model.fit_generator(train_generator,
                              epochs=26,callbacks=[callbacks],
                              verbose=1,
                              validation_data=validation_generator)



