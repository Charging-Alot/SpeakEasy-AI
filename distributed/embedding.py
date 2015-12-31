from __future__ import absolute_import
# from __future__ import division
from __future__ import print_function

import sys
import os
path = os.path.join(os.path.dirname(__file__), '..') 
sys.path.append(path)

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import random
import linecache

import tensorflow as tf
from flask import Flask, jsonify, make_response, request, abort 
import json

from data import data_utils
from model.train import read_data
from runtime_variables import params, buckets

# application = Flask(__name__)

ENCODER_EMBEDDING = np.load('/Users/lauragelston/Desktop/encoder_embedding.npy')
DECODER_EMBEDDING = np.load('/Users/lauragelston/Desktop/decoder_embedding.npy')
# train_set = None

# def initialize():
  # global train_set
  # if not train_set:
  #   train_set, _ = read_data(params.data_dir + '/BABYTRAIN_25000.ids25000', 20)
  #   print ("Data loaded")
    

def encode_encoder_input(encoder_inputs):
  return tf.nn.embedding_lookup(ENCODER_EMBEDDING, tf.reshape(encoder_inputs, [-1]))

def encode_decoder_input(decoder_inputs):
  return tf.nn.embedding_lookup(DECODER_EMBEDDING, tf.reshape(decoder_inputs, [-1]))

def extract_argmax(output_logits):
  outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]

def make_batch():
  encoder_input, decoder_input = read_data()
  print(encoder_input, decoder_input)
  encoder_size, decoder_size = len(encoder_input), len(decoder_input)



  encoder_inputs, decoder_inputs = [], []
  encoder_inputs.append(list(reversed(encoder_input)))
  decoder_inputs.append([data_utils.GO_ID] + decoder_input)

  batch_encoder_inputs, batch_decoder_inputs = [], []
  for length_idx in xrange(encoder_size):
    batch_encoder_inputs.append(np.array([encoder_inputs[0][length_idx]], dtype=np.int32))
  for length_idx in xrange(decoder_size):
    batch_decoder_inputs.append(np.array([decoder_inputs[0][length_idx]], dtype=np.int32))

  return batch_encoder_inputs, batch_decoder_inputs

def run_tensors(embedded_encoders, embedded_decoders):
  print('Running tensors')
  encoder_sess = tf.Session()
  decoder_sess = tf.Session()
  encoder = [encoder_sess.run(embedded_encoders[i]).tolist() for i in xrange(len(embedded_encoders))]
  decoder = [decoder_sess.run(embedded_decoders[j]).tolist() for j in xrange(len(embedded_decoders))]
  encoder_sess.close()
  decoder_sess.close()
  return encoder, decoder

# @application.route('/data', methods=["GET"])
def embed():
  data_id = (params.readline + 1) / 2 
  encoder_inputs, decoder_inputs = make_batch()
  embedded_encoders = [encode_encoder_input(encoder_inputs[i]) for i in xrange(len(encoder_inputs))]
  embedded_decoders = [encode_decoder_input(decoder_inputs[j]) for j in xrange(len(decoder_inputs))]
  try:
    print('Ran tensors!')
    encoder, decoder = run_tensors(embedded_encoders, embedded_decoders)
  except ValueError:
    print("OH NO", data_id)
    encoder, decoder = "Session overloaded", "Session overloaded"

  with open('/Volumes/HD/SPEAKEASY_DATA/REDDIT/reddit_data/embedded/%d'%data_id, 'w') as encoder_file:
    encoder_file.write(json.dumps({"encoder" : encoder, "decoder" : decoder}))
  
  # train_bucket_sizes = [len(train_set[b]) for b in xrange(len(buckets))]
  # train_total_size = float(sum(train_bucket_sizes))
  # train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size for i in xrange(len(train_bucket_sizes))]

  # random_number_01 = np.random.random_sample()
  # bucket_id = min([i for i in xrange(len(train_buckets_scale)) if train_buckets_scale[i] > random_number_01])
  # bucket_id = 0
  # encoder_inputs, decoder_inputs = get_batch(train_set, bucket_id=bucket_id)
    

  # print(json.dumps({"encoder" : encoder, "decoder" : decoder}), "UGH")
  # with open('/Users/lauragelston/Desktop/encoder', 'w') as encoder_file:
  #   encoder_file.write(json.dumps(encoder))
  # with open('/Users/lauragelston/Desktop/decoder', 'w') as decoder_file:
  #   decoder_file.write(json.dumps(decoder))
  # test_file.close()
  # return make_response(jsonify({'encoder': encoder, 'decoder' : decoder}), 200)
def read_data():
  prompt = linecache.getline(params.data_dir + '/BABYTRAIN_25000.ids25000', params.readline)
  response = linecache.getline(params.data_dir + '/BABYTRAIN_25000.ids25000', params.readline + 1)
  prompt_ids = [int(x) for x in prompt.split()]
  response_ids = [int(x) for x in response.split()]
  response_ids.append(data_utils.EOS_ID)

  return prompt_ids, response_ids


if __name__ == '__main__':
  # initialize()
  embed()
#   application.run(debug=True)
# else:
#   Thread(target=initialize).start()
