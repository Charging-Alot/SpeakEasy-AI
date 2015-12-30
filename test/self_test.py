"""Test the training protocol."""
from __future__ import print_function
from __future__ import absolute_import

import sys
import os
path = os.path.join(os.path.dirname(__file__), '..') 
sys.path.append(path) 

import tensorflow as tf
import random

from model.seq2seq_model import Seq2SeqModel
from runtime_variables import params, buckets

def self_test():
  with tf.Session() as sess:
    print("Self-test for neural translation model.")
    # Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
    model = Seq2SeqModel(10, [(3, 3), (6, 6)], 32, 2, 5.0, 32, 0.3, 0.99, params.model_type, num_samples=8)
    sess.run(tf.initialize_all_variables())
    # Fake data set for both the (3, 3) and (6, 6) bucket.
    data_set = ([([1, 1], [2, 2]), ([3, 3], [4]), ([5], [6])], [([1, 1, 1, 1, 1], [2, 2, 2, 2, 2]), ([3, 3, 3], [5, 6])])
    for _ in xrange(5):  # Train the fake model for 5 steps.
      if params.buckets:
        bucket_id = random.choice([0, 1])
        encoder_inputs, decoder_inputs, target_weights = model.get_batch(data_set, bucket_id=bucket_id)
        print(encoder_inputs, "AFTER BATCH")
        model.step(sess, encoder_inputs, decoder_inputs, target_weights, True, bucket_id=bucket_id)
      else:
        encoder_inputs, decoder_inputs, target_weights = model.get_batch(data_set, bucket_id=None)
        model.step(sess, encoder_inputs, decoder_inputs, target_weights, True, bucket_id=None)
    print('Test successful!')
