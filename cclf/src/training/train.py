from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Reshape

from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential

import tensorflow as tf
from PIL import Image

src_path_train = "../data/train/"
src_path_test = "../data/test/"


COLOUR_MODE = "rgb"
BATCH_SIZE = 8
CLASS_MODE = "categorical"
TARGET_SIZE = (224, 224)

assert tf.config.list_physical_devices('GPU')

train_data = tf.keras.preprocessing.image_dataset_from_directory(src_path_train,
                                                                                label_mode="categorical",
                                                                                image_size=TARGET_SIZE)
                                                                                
test_data = tf.keras.preprocessing.image_dataset_from_directory(src_path_test,
                                                                label_mode="categorical",
                                                                image_size=TARGET_SIZE,
                                                                shuffle=False)

# Setup data augmentation
data_augmentation = Sequential([
  preprocessing.RandomFlip("horizontal"), # randomly flip images on horizontal edge
  preprocessing.RandomRotation(0.2), # randomly rotate images by a specific amount
  preprocessing.RandomHeight(0.2), # randomly adjust the height of an image by a specific amount
  preprocessing.RandomWidth(0.2), # randomly adjust the width of an image by a specific amount
  preprocessing.RandomZoom(0.2), # randomly zoom into an image
  preprocessing.Rescaling(1./255) 
], name="data_augmentation")

model = tf.keras.applications.EfficientNetB0(include_top=False)

inputs = layers.Input(shape=(224, 224, 3), name="input_layer") # shape of input image
# x = data_augmentation(inputs) # augment images (only happens during training)
x = model(inputs, training=False) # put the base model in inference mode so we can use it to extract features without updating the weights
x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x) # pool the outputs of the base model
outputs = layers.Dense(len(train_data.class_names), activation="softmax", name="output_layer")(x) # same number of outputs as classes
model = tf.keras.Model(inputs, outputs)

# Compile
model.compile(loss="categorical_crossentropy",
              optimizer=tf.keras.optimizers.Adam(), # use Adam with default settings
              metrics=["accuracy"])
model.fit(train_data,
                                           epochs=5, # fit for 5 epochs to keep experiments quick
                                           validation_data=test_data,
                                           validation_steps=int(0.15 * len(test_data)), # evaluate on smaller portion of test data
                                           ) # save best model weights to fil