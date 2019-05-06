from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras import backend as K
from keras.layers.normalization import BatchNormalization

class TennisVGGNet:
    @staticmethod
    def build(width, height, depth, classes):
        input_shape = (height, width, depth)
        chanDim = -1
        if K.image_data_format() == "channels_first":
	    inputShape = (depth, height, width)
	    chanDim = 1

        model = Sequential()

        model.add(Conv2D(64, (3, 3), input_shape=input_shape, padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))

        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (3, 3),padding='same'))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))

        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(Activation("relu"))
	model.add(BatchNormalization(axis=chanDim))

        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(256, (3, 3), padding='same'))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))

        model.add(Conv2D(256, (3, 3), padding='same'))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))

        model.add(Conv2D(256, (3, 3), padding='same'))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))

        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization(axis=chanDim))

        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization(axis=chanDim))

        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization(axis=chanDim))

        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization(axis=chanDim))

        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization(axis=chanDim))

        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization(axis=chanDim))

        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(256, activation='relu',name = "fc"))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))
        model.add(Dense(classes, activation='softmax'))
        model.summary()
        return model
