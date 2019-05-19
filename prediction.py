import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# print(keras.__version__)
# print(tf.__version__)

train_causes = []
train_result = []
test_causes = []
# model = keras.Sequential()
# graph = tf.Graph()

# READ------------------------------------------------------------------------------------------------------------------------------------------------

def read_training_set():
    global train_causes, train_result
    trainSet = pd.read_csv('/home/nonetto/catkin_ws/src/nonettobot/src/train_dataset.csv')
    trainSet = np.array(trainSet)

    for i in range(len(trainSet)):
        # if trainSet[i][0] == 0:
        train_causes.append(trainSet[i][0:8])
        train_result.append(trainSet[i][8:10])

    train_causes = np.array(train_causes)
    train_result = np.array(train_result)

# TRAIN------------------------------------------------------------------------------------------------------------------------------------------------

def train_model():
    # global model, graph
    # c = tf.constant(1.0, dtype=tf.float64)
    alpha=tf.constant(0.2, dtype=tf.float64)

    model = keras.Sequential([
        keras.layers.Dense(64, kernel_initializer=keras.initializers.glorot_normal(seed=None), kernel_regularizer=keras.regularizers.l2(0.0001), activation='tanh'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(16, activation='linear'),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Dense(2)
    ])

    opt = keras.optimizers.Adam(lr=0.0001, decay=0, amsgrad=False)

    model.compile(optimizer=opt, loss="mean_squared_error")
    model.fit(train_causes, train_result, validation_split=0.2, epochs=500, batch_size=10)#, verbose=0)
    # model.summary()
    # model.save('last_location_prediction_model.h5')
    keras.experimental.export_saved_model(model, 'last_location_prediction_model')
    # return model
    # model._make_predict_function()
    # graph = tf.get_default_graph()
    # model.save_weights("model_weights.h5")

# PREDICT------------------------------------------------------------------------------------------------------------------------------------------------    

def predict(causes):
    # global model, graph
    model = keras.experimental.load_from_saved_model('last_location_prediction_model')
    print("model loaded")
    causes = np.array(causes)
    # causes = np.transpose(causes)
    causes = np.reshape(causes, (1,8))
    # print(causes.shape)

    # with graph.as_default():
    predictions = model.predict(causes, batch_size=1, verbose=0)
    return predictions
    #     return predictions

# --------------------------------------------------------------------------------------------------------------------------------------------------------

def main():
    read_training_set()
    print ("read training set")
    train_model()
    print("trained")

if __name__ == '__main__':
    main()