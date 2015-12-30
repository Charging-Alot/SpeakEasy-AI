from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
path = os.path.join(os.path.dirname(__file__), '..') 
sys.path.append(path)

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import random

import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import seq2seq

from data import data_utils
from model.model_utils import create_model
from model.train import read_data
from runtime_variables import params, buckets

ENCODER_EMBEDDING = np.load('/Users/lauragelston/Desktop/encoder_embedding.npy')
DECODER_EMBEDDING = np.load('/Users/lauragelston/Desktop/decoder_embedding.npy')

def encode_encoder_input(encoder_inputs):
  return tf.nn.embedding_lookup(ENCODER_EMBEDDING, tf.reshape(encoder_inputs, [-1]))

def encode_decoder_input(decoder_inputs):
  return tf.nn.embedding_lookup(ENCODER_EMBEDDING, tf.reshape(decoder_inputs, [-1]))

def extract_argmax(output_logits):
  outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]

def main():
  sess = tf.Session()
  model = create_model(sess, False)
  model.batch_size = 1
  # data_set, epoch = read_data(params.data_dir + 'BABYTRAIN25000.ids25000')

  train_set = ([([1, 1], [2, 2]), ([3, 3], [4]), ([5], [6])], [([1, 1, 1, 1, 1], [2, 2, 2, 2, 2]), ([3, 3, 3], [5, 6])])


  # train_bucket_sizes = [len(train_set[b]) for b in xrange(len(buckets))]
  # train_total_size = float(sum(train_bucket_sizes))
  # train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size for i in xrange(len(train_bucket_sizes))]

  # random_number_01 = np.random.random_sample()
  # bucket_id = min([i for i in xrange(len(train_buckets_scale)) if train_buckets_scale[i] > random_number_01])

  bucket_id = 1
  encoder_inputs, decoder_inputs, target_weights = model.get_batch(train_set, bucket_id=bucket_id)
  print (type(encoder_inputs), "WHAT THE FUCK")
  print(encoder_inputs)
  print(encoder_inputs[0].shape)
  print(encoder_inputs[1])
  print(encoder_inputs[1].shape)

  embedded_encoders = [encode_encoder_input(encoder_inputs[i]) for i in xrange(len(encoder_inputs)-1)]
  embedded_decoders = [encode_decoder_input(decoder_inputs[j]) for j in xrange(len(decoder_inputs)-1)]

  print(type(embedded_encoders))
  print(embedded_encoders[0])
  print(embedded_decoders)

if __name__ == "__main__":
  main()

