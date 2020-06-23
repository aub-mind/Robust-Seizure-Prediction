"""
This file is used to check the generated AEs

"""


import numpy as np
import tensorflow as tf
from models.helping_functions import  data_loading, generate_adversarial
from models.model_ae import CNN_GRU
from keras.utils import to_categorical


# inspect AEs
dataset ='CHBMIT'
ictal_X, ictal_y ,interictal_X, interictal_y, settings = data_loading(target='1', dataset=dataset)



validation = "test" 
tf.reset_default_graph()
graph = tf.get_default_graph()
session = tf.Session()


noise_limit = 0.3 #   of the max value
model = CNN_GRU([ictal_X.shape[2], ictal_X.shape[3]], dataset, noise_limit, graph)  #Build the graph
model.train_eval_test_ae(session, '2', ictal_X, ictal_y ,interictal_X, interictal_y, 
                         settings, validation, mode='without_AE', batch_size=256, epoch=20, percentage=0.1)

preictal = np.array([[0, 1]])
interictal = np.array([1, 0])
input_eeg = np.concatenate((interictal_X.reshape(-1, ictal_X.shape[2], ictal_X.shape[3]),
                            ictal_X.reshape(-1, ictal_X.shape[2], ictal_X.shape[3])))
input_labeles = to_categorical(np.concatenate((interictal_y.reshape(-1), ictal_y.reshape(-1))))

# visualize the generated AE with the coresponding input and model prediction
generate_adversarial(model, session, preictal, input_eeg, input_labeles, 100)