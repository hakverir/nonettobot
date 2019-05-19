import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
# from keras.backend import clear_session

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

train_causes_sphere = []
train_result_sphere = []
test_causes_sphere = []
model_sphere_x = keras.Sequential()
model_sphere_y = keras.Sequential()
predictions_x = []
predictions_y = []

def read_training_set():
    global train_causes_sphere, train_result_sphere
    trainSet = pd.read_csv('/home/nonetto/catkin_ws/src/nonettobot/src/train_dataset.csv')
    trainSet = np.array(trainSet)

    for i in range(len(trainSet)):
        if trainSet[i][0] == 0:
            train_causes_sphere.append(trainSet[i][1:8])
            train_result_sphere.append(trainSet[i][8:10])

    train_causes_sphere = np.array(train_causes_sphere)
    train_result_sphere = np.array(train_result_sphere)

def train_x():
    global model_sphere_x
    #, kernel_initializer=kernel_init kernel_regularizer=keras.regularizers.l2(0.0001),
    model_sphere_x = keras.Sequential([
        keras.layers.Dense(16, kernel_initializer=keras.initializers.glorot_normal, kernel_regularizer=keras.regularizers.l2(0.0001), activation=tf.nn.tanh),
        # keras.layers.BatchNormalization(),
        # keras.layers.Dropout(0.01, noise_shape=None, seed=None),
        keras.layers.Dense(8, activation=tf.nn.relu),
        keras.layers.Dense(4, activation=tf.nn.leaky_relu),
        keras.layers.Dense(1)
    ])

    opt = keras.optimizers.Adam(lr=0.0001, decay=0.0001, amsgrad=False)

    model_sphere_x.compile(optimizer=opt, loss="mean_squared_error")

    history = model_sphere_x.fit(train_causes_sphere[:, 0:6], train_result_sphere[:, 0:1], validation_split=0.2,
                                 epochs=100, batch_size=10, verbose=0)
    return history

def train_y():
    global model_sphere_y

    model_sphere_y = keras.Sequential([
        keras.layers.Dense(16, kernel_initializer=keras.initializers.glorot_normal, kernel_regularizer=keras.regularizers.l2(0.0001), activation=tf.nn.tanh), # kernel_regularizer=keras.regularizers.l2(0.0001)
        # keras.layers.BatchNormalization(),
        # keras.layers.Dropout(0.01, noise_shape=None, seed=None),
        keras.layers.Dense(8, activation=tf.nn.relu),
        keras.layers.Dense(4, activation=tf.nn.leaky_relu),
        keras.layers.Dense(1)
    ])

    opt = keras.optimizers.Adam(lr=0.0001, decay=0.0001, amsgrad=False)

    model_sphere_y.compile(optimizer=opt, loss="mean_squared_error")

    train_causes_sphere_y = np.concatenate((train_causes_sphere[:, 0:5], train_causes_sphere[:, 6:7]), axis=1)

    history = model_sphere_y.fit(train_causes_sphere_y, train_result_sphere[:, 1:2], validation_split=0.2,
                                 epochs=100, batch_size=10, verbose=0)
    return history

def predict(causes, model_x):#, model_y):
    # keras.backend.clear_session()
    # test_causes_sphere = causes
    causes = np.array(causes)
    # test_sph_coo_y = np.concatenate((causes[:, 0:5], causes[:, 6:7]), axis=1)
    test_sph_coo_y = causes[[0, 1, 2, 3, 4, 6]]

    # global predictions_x, predictions_y
    # model_x._make_predict_function()
    try:
        predictions_x = model_x.predict(causes[[0, 1, 2, 3, 4, 5]], batch_size=10, verbose=0)
    except:
        import pdb, traceback, sys
        extype,value,tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
    # model_y._make_predict_function()
    # predictions_y = model_y.predict(test_sph_coo_y, batch_size=10, verbose=0)
    # return [predictions_x, predictions_y]
    return predictions_x

def main():
    global model_sphere_x, model_sphere_y, test_causes_sphere
    read_training_set()

    print("read the training set, waiting for the training")

    hist_x = train_x()
    print("trained x")
    # hist_y = train_y()

    print("training finished")
    # predict()

    # global predictions_x, predictions_y
    # return [predictions_x, predictions_y]
    # return [model_sphere_x, model_sphere_y]
    return model_sphere_x

if __name__ == "__main__":
    main()
