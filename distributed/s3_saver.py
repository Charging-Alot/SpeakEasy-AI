from __future__ import absolute_import
from __future__ import print_function

import sys
import os
path = os.path.join(os.path.dirname(__file__), '..') 
sys.path.append(path)

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import shutil
import boto3
s3 = boto3.resource('s3')

from runtime_variables import params
from model.model_utils import create_model

SAVE_DIR = '/Volumes/HD/SPEAKEASY_DATA/REDDIT/'

VARIABLES = [
    "learning_rate", 
    "global_step", 
    "projW", 
    "projB", 
    "encoder_embedding", 
    "LSTM_encoder1_weights", 
    "LSTM_encoder1_biases", 
    "LSTM_encoder2_weights", 
    "LSTM_encoder2_biases",
    "LSTM_encoder3_weights",
    "LSTM_encoder3_biases",
    "decoder_embedding",
    "LSTM_decoder1_weights", 
    "LSTM_decoder1_biases", 
    "LSTM_decoder2_weights", 
    "LSTM_decoder2_biases",
    "LSTM_decoder3_weights",
    "LSTM_decoder3_biases",
    ]

def save_variables():
  with tf.Session() as sess:
    model = create_model(sess, True)
    model_params = sess.run(tf.all_variables())
    for i in xrange(2,len(VARIABLES)):
      print(VARIABLES[i])
      np.savetxt(SAVE_DIR + VARIABLES[i], model_params[i])
      upload_variable(VARIABLES[i])


def upload_variable(variable):
  s3.meta.client.upload_file(SAVE_DIR +  variable, 'speakeasy-data', 'trainable_variables/' + variable)

if __name__ == "__main__":
  print('Saving to s3')
  save_variables()
