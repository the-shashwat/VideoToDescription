from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam,SGD
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tennisVGGModel import TennisVGGNet
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import random
import pickle
import cv2
import os
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "5"
session = tf.Session(config=config)
from keras.backend.tensorflow_backend import set_session
set_session(session)

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
epochs = 500
learning_rate = 0.01
batch_size = 16
image_dims = (224,224,3)
momentum = 0.5
decay = 1e-6

label_binarizer_path = "./lbl_bin/lb.pickle"
models_dir = "./flow-cnn-models4/"
image_dir = "./data/flow_dataset3/"
data = []
labels = []
model_save_path = os.path.join(models_dir,"cnn-final-model.model")
# grab the image paths and randomly shuffle them
print("[INFO] loading images...")

'''paths = []
with open("folder_list.txt","r") as fd:
    for x in fd.readlines():
        paths.append(x[0:-1])

paths = sorted(paths)
random.seed(42)
random.shuffle(paths)

for path in paths:
    img_list = os.listdir(path)
    # extract the class label from the image path and update the
    # labels list
    label = path.split(os.path.sep)[-2]
    labels.append(label)
    image_stack = np.zeros(shape=(image_dims[1], image_dims[0], image_dims[2]))
    for i,img in enumerate(img_list):
        image = cv2.imread(os.path.join(path,img))
        image = cv2.resize(image, (image_dims[1], image_dims[0]), interpolation = cv2.INTER_AREA)
        image = img_to_array(image) 
        image_stack[:, :, 3*i:3*i+3 ] = image
    data.append(image_stack)

'''
imagePaths = sorted(list(paths.list_images(image_dir)))
random.seed(42)
random.shuffle(imagePaths)
cnt = -1
for imagePath in imagePaths:
    # extract the class label from the image path and update the
    # labels list
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (image_dims[1], image_dims[0]), interpolation = cv2.INTER_AREA)
    image = img_to_array(image)
    data.append(image)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
 
# binarize the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.2, random_state=42)

# save the label binarizer to disk
print("[INFO] serializing label binarizer...")
f = open(label_binarizer_path, "wb")
f.write(pickle.dumps(lb))
f.close()

# construct the image generator for data augmentation
#aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
#	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
#	horizontal_flip=True, fill_mode="nearest")

# initialize the model
print("[INFO] compiling model...")
model = TennisVGGNet.build(width=image_dims[1], height=image_dims[0],
	depth=image_dims[2], classes=len(lb.classes_))

#model.load_weights("./flow-cnn-models/weights-improvement-74-0.41.hdf5")
#opt = Adam(lr=learning_rate, decay=learning_rate / epochs)
opt = SGD(lr=learning_rate, momentum=momentum, decay=decay, )
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])
 
filepath=os.path.join(models_dir,"weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5")
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only = False, mode = 'max', period = 10)
callbacks_list = [checkpoint]
# train the network
print("[INFO] training network...")
#H = model.fit_generator(
#	trainX, trainY, batch_size=batch_size),
#	validation_data=(testX, testY),
#	steps_per_epoch=len(trainX) // batch_size,
#	epochs=epochs, verbose=1, callbacks=callbacks_list)
H = model.fit(x = trainX, y = trainY, 
        batch_size = batch_size, epochs = epochs, 
        callbacks=callbacks_list, 
        validation_data=(testX, testY), 
        verbose = 1)
# save the model to disk
print("[INFO] serializing network...")
model.save(model_save_path)

''' 
# save the label binarizer to disk
print("[INFO] serializing label binarizer...")
f = open(label_binarizer_path, "wb")
f.write(pickle.dumps(lb))
f.close()
'''
