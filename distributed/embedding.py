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
from flask import Flask, jsonify, make_response, request, abort 

from data import data_utils
from model.train import read_data
from runtime_variables import params, buckets

application = Flask(__name__)

ENCODER_EMBEDDING = np.load('/Users/lauragelston/Desktop/encoder_embedding.npy')
DECODER_EMBEDDING = np.load('/Users/lauragelston/Desktop/decoder_embedding.npy')
train_set = None

def initialize():
  global train_set
  if not train_set:
    train_set, _ = read_data(params.data_dir + '/BABYTRAIN_25000.ids25000', 20)
    print ("Data loaded")
    

def encode_encoder_input(encoder_inputs):
  return tf.nn.embedding_lookup(ENCODER_EMBEDDING, tf.reshape(encoder_inputs, [-1]))

def encode_decoder_input(decoder_inputs):
  return tf.nn.embedding_lookup(ENCODER_EMBEDDING, tf.reshape(decoder_inputs, [-1]))

def extract_argmax(output_logits):
  outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]

def get_batch(data, bucket_id=None):
  encoder_inputs, decoder_inputs = [], []
  # Get a random tuple of data
  # pad if needed, reverse encoder input and add GO to decoder.
  encoder_input, decoder_input = random.choice(data[bucket_id])
  encoder_size, decoder_size = len(encoder_input), len(decoder_input)
  encoder_inputs.append(list(reversed(encoder_input)))

  # Decoder inputs get an extra "GO" symbol, and are padded then.
  decoder_pad_size = decoder_size - len(decoder_input) - 1
  decoder_inputs.append([data_utils.GO_ID] + decoder_input)

  # Now we create batch-major vectors from the data selected above.
  batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

  # Batch encoder inputs are just re-indexed encoder_inputs.
  for length_idx in xrange(encoder_size):
    batch_encoder_inputs.append(np.array([encoder_inputs[0][length_idx]], dtype=np.int32))
  print(batch_encoder_inputs)

  # Batch decoder inputs are re-indexed decoder_inputs.
  for length_idx in xrange(decoder_size):
    batch_decoder_inputs.append(np.array([decoder_inputs[0][length_idx]], dtype=np.int32))

  return batch_encoder_inputs, batch_decoder_inputs
def run_tensors(embedded_encoders, embedded_decoders):
  encoder_sess = tf.Session()
  decoder_sess = tf.Session()
  print (tf.all_variables(), "VARIABLES")
  encoder = [encoder_sess.run(embedded_encoders[i]).tolist() for i in xrange(len(embedded_encoders))]
  decoder = [decoder_sess.run(embedded_decoders[j]).tolist() for j in xrange(len(embedded_decoders))]
  encoder_sess.close()
  decoder_sess.close()
  return encoder, decoder

@application.route('/data', methods=["GET"])
def embed():
  if not train_set:
    print("Data not yet loaded")
    abort(418)

  train_bucket_sizes = [len(train_set[b]) for b in xrange(len(buckets))]
  train_total_size = float(sum(train_bucket_sizes))
  train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size for i in xrange(len(train_bucket_sizes))]

  random_number_01 = np.random.random_sample()
  bucket_id = min([i for i in xrange(len(train_buckets_scale)) if train_buckets_scale[i] > random_number_01])
  bucket_id = 0
  encoder_inputs, decoder_inputs = get_batch(train_set, bucket_id=bucket_id)
    
  embedded_encoders = [encode_encoder_input(encoder_inputs[i]) for i in xrange(len(encoder_inputs))]
  embedded_decoders = [encode_decoder_input(decoder_inputs[j]) for j in xrange(len(decoder_inputs))]
  encoder, decoder = run_tensors(embedded_encoders, embedded_decoders)

  print(embedded_encoders)
  print(embedded_encoders[0])
  
  
  return make_response(jsonify({'encoder': encoder, 'decoder' : decoder}), 200)
  
if __name__ == '__main__':
  initialize()
  application.run(debug=True)
else:
  Thread(target=initialize).start()
