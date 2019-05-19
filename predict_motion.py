import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# PREDICT------------------------------------------------------------------------------------------------------------------------------------------------    

def predict(causes):
    model = keras.experimental.load_from_saved_model('last_location_prediction_model')

    causes = np.array(causes)
    causes = np.reshape(causes, (1,8))

    predictions = model.predict(causes, batch_size=1, verbose=0)
    predictions = predictions.tolist()
    return predictions[0]

# --------------------------------------------------------------------------------------------------------------------------------------------------------

def main(causes):
    return predict(causes)

if __name__ == '__main__':
    main()