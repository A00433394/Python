"""
@author : Haozhou Wang(A00431268)
          Rishi Karki (A00432524)
          Hemanchal Joshi (A00433394)

This is a multiple-bael fruit classifier programme based on keras
"""

import glob
import os

from keras import backend as K
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Activation
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.optimizers import rmsprop
from keras import utils

import cv2
import numpy as np
from matplotlib import pyplot as plt


def get_image(label, x, y):
    for i in range(0, len(y)):
        if np.argmax(y[i]) == label:
            return x[i]
    return None


def image_subset(index, x, y):
    """
    index equals to amount of labels
    """
    xs = []
    ys = []
    print("len(x) is {}".format(len(x)))
    for i in range(len(x)):
        if y[i] < index:
            xs.append(x[i])
            ys.append(y[i])
    return np.array(xs), np.array(ys)


def get_images(path_list):
    """
    read image dataset based on given path
    """
    images = []
    labels = []
    names = []
    i = 0
    print(path_list)
    for path in path_list:
        for fruit_dir_path in glob.glob(path):
            fruit_label = fruit_dir_path.split("/")[-1]
            for image_path in glob.glob(os.path.join(fruit_dir_path, "*.jpg")):
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)

                image = cv2.resize(image, (45, 45))
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                images.append(image)
                names.append(fruit_label)
                labels.append(i)
            i += 1

    images = np.array(images)
    print(images.shape)
    # add a new dimension here
    with np.nditer(images, op_flags=['readwrite']) as it:
        for x in it:
            x = np.expand_dims(x, axis=0)
    labels = np.array(labels)
    return images, labels, i


class CNN(object):
    """
    A convolutional neural network
    """

    def __init__(self, train_directory, test_directory, num_classes, batch_size, epochs):
        self.train_directory = train_directory
        self.test_directory = test_directory
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.epochs = epochs
        self.build_dataset()

    def build_dataset(self):
        """
        split it into proper train/test dataset
        """
        print("reading data of images currently , please wait......")
        x_train, y_train, _ = get_images(self.train_directory)
        x_test, y_test, _ = get_images(self.test_directory)
        x_train, y_train = image_subset(self.num_classes, x_train, y_train)
        x_test, y_test = image_subset(self.num_classes, x_test, y_test)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        self.x_train = x_train / 255
        self.x_test = x_test / 255
        self.y_train = utils.to_categorical(y_train, self.num_classes)
        self.y_test = utils.to_categorical(y_test, self.num_classes)

    def build_model(self, method):
        if method == 1:
            model = Sequential()
            model.add(Conv2D(32, (3, 3), activation='relu',
                             padding='same',
                             input_shape=(45, 45, 3)))
            model.add(Conv2D(32, (3, 3),
                             activation='relu'))
            model.add(MaxPooling2D((2, 2)))
            model.add(Dropout(0.1))
            model.add(Conv2D(64, (3, 3),
                             padding='same',
                             activation='relu'))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D((2, 2)))
            model.add(Dropout(0.25))
            model.add(Flatten())
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.25))
            model.add(Dense(num_classes, activation='softmax'))
            es = EarlyStopping(monitor='val_loss',
                               patience=15,
                               min_delta=0.01)
            opt = rmsprop(lr=0.0001, decay=1e-6)
            model.compile(optimizer=opt,
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
            self.hist = model.fit(self.x_train, self.y_train,
                                  batch_size=self.batch_size,
                                  epochs=self.epochs,
                                  validation_data=(
                                      self.x_test, self.y_test),
                                  callbacks=[es],
                                  shuffle=True,
                                  verbose=2)
            # hard-coded name of model saved for convenience
            model.save("assignment.h5")
        else:
            print("load model from h5 .......")
            model = load_model('Assignment.h5')
        score = model.evaluate(self.x_test, self.y_test)
        print('Test loss: {0}'.format(score[0]))
        print('Test accuracy: {0}'.format(score[1]))

    def draw_plots(self):
        plt.plot(self.hist.history['val_loss'], label="val_loss")
        plt.plot(self.hist.history['val_acc'], label="val_acc")
        plt.plot(self.hist.history['loss'], label="loss")
        plt.plot(self.hist.history['acc'], label="acc")
        plt.xlabel("epoch")
        plt.ylabel("metrics")
        plt.title("trendency")
        plt.ylim(ymin=0)
        plt.legend()
        plt.show()


if __name__ == '__main__':
    """
    configure variable
    To run the code in your laotopb , maybe you need to change some variables like directory.
    """

    num_classes = 4
    train_directory = ["data/fruits-360/Training/Apricot", "data/fruits-360/Training/Avocado",
                       "data/fruits-360/Training/Banana", "data/fruits-360/Training/Cherry", ]
    test_directory = ["data/fruits-360/Validation/*"]

    batch_size = 32
    epochs = 10

    # if you want to build the model from scratch , use 1 as paramater; If you want to load h5 model ,pass 2 as paramater.
    cnn = CNN(train_directory, test_directory, num_classes, batch_size, epochs)

    cnn.build_model(1)
    cnn.draw_plots()
    
    """
    We can't draw the plots if we get the model from saved h5 file 
    There will be no history object
    reference : https://stackoverflow.com/questions/47843265/how-can-i-get-the-a-keras-models-history-after-loading-it-from-a-file-in-python
    """
    

