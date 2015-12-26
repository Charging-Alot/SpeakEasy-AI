"""Sequence-to-sequence model with an attention mechanism."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.framework import ops

from rnn import rnn_cell
from rnn import seq2seq

import data_utils

class Seq2SeqModel(object):
  def __init__(self, vocab_size, max_sentence_length, size, num_layers, max_gradient_norm, batch_size, learning_rate, learning_rate_decay_factor, num_samples=512, forward_only=False, model_type):

    self.vocab_size = vocab_size
    self.batch_size = batch_size
    self.max_sentence_length = max_sentence_length
    self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
    self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
    self.global_step = tf.Variable(0, trainable=False)

    # output projection for sampled softmax:
    output_projection = None
    softmax_loss_function = None

    if num_samples > 0 and num_samples < self.vocab_size:
      with tf.device("/cpu:0"):
        w = tf.get_variable("proj_w", [size, self.vocab_size])
        w_t = tf.transpose(w)
        b = tf.get_variable("proj_b", [self.vocab_size])
      output_projection = (w, b)

      def sampled_loss(inputs, labels):
        with tf.device("/cpu:0"):
          labels = tf.reshape(labels, [-1, 1])
          return tf.nn.sampled_softmax_loss(w_t, b, inputs, labels, num_samples, self.vocab_size)
      softmax_loss_function = sampled_loss

    # single LSTM cell creation, use to build hidden layers
    single_cell=rnn_cell.BasicLSTMCell(size)
    cell = rnn_cell.MultiRNNCell([single_cell] * num_layers)
    
    if model_type == 'embedding_attention':
      def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
        return seq2seq.embedding_attention_seq2seq(encoder_inputs, decoder_inputs, cell, vocab_size, vocab_size, output_projection=output_projection, feed_previous=do_decode)
    else:
      def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
        return seq2seq.embedding_rnn_seq2seq(encoder_inputs, decoder_inputs, cell, vocab_size, vocab_size, output_projection=output_projection, feed_previous=do_decode)

    # feeds for inputs are limited to max_sentence_length
    self.encoder_inputs = []
    self.decoder_inputs = []
    self.target_weights = []
    for i in xrange(max_sentence_length):
      self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="encoder{0}".format(i)))
    for i in xrange(max_sentence_length + 1):
      self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="decoder{0}".format(i)))
      self.target_weights.append(tf.placeholder(tf.float32, shape=[None], name="weight{0}".format(i)))

    # targets are decoder inputs shifted by one
    targets = [self.decoder_inputs[i + 1]
               for i in xrange(len(self.decoder_inputs) - 1)]

    # training outputs and losses
    if forward_only:
      self.outputs, self.states = seq2seq_f(self.encoder_inputs, self.decoder_inputs[:-1], True)
      self.losses = seq2seq.sequence_loss(self.outputs, targets, self.target_weights[:-1], self.vocab_size, softmax_loss_function=softmax_loss_function)
      # project outputs for decoding
      if output_projection is not None:
        self.outputs = [tf.nn.xw_plus_b(output, output_projection[0], output_projection[1]) for output in self.outputs]
    else:
      self.outputs, self.states = seq2seq_f(self.encoder_inputs, self.decoder_inputs[:-1], False)
      self.losses = (seq2seq.sequence_loss(self.outputs, targets, self.target_weights[:-1], self.vocab_size, softmax_loss_function=softmax_loss_function))

    # gradients and SGD update operation for training
    params = tf.trainable_variables()
    self.params = params
    if not forward_only:
      self.gradient_norms = []
      self.updates = []
      opt = tf.train.GradientDescentOptimizer(self.learning_rate)

      gradients = tf.gradients(self.losses, params)
      clipped_gradients, norm = tf.clip_by_global_norm(gradients,max_gradient_norm)
      self.gradient_norms = norm
      self.updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

    self.saver = tf.train.Saver(tf.all_variables())    

  def step(self, session, encoder_inputs, decoder_inputs, target_weights,forward_only):
    encoder_size, decoder_size = self.max_sentence_length, self.max_sentence_length
    if len(encoder_inputs) != encoder_size:
      raise ValueError("Encoder length must be equal to max sentence length")
    if len(decoder_inputs) != decoder_size:
        raise ValueError("Decoder length must be equal to max sentence length")
    if len(target_weights) != decoder_size:
        raise ValueError("Target weights must be equal to max sentence length")

    # input feed: encoder inputs, decoder inputs, target_weights, as provided.
    input_feed = {}
    for l in xrange(encoder_size):
      input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
    for l in xrange(decoder_size):
      input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
      input_feed[self.target_weights[l].name] = target_weights[l]

    # since our targets are decoder inputs shifted by one, we need one more.
    last_target = self.decoder_inputs[decoder_size].name
    input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

    # Output feed: depends on whether we do a backward step or not.
    if not forward_only:
      output_feed = [self.updates,  # Update Op that does SGD.
                     self.gradient_norms,  # Gradient norm.
                     self.losses]  # Loss for this batch.
    else:
      output_feed = [self.losses]  # Loss for this batch.
      for l in xrange(decoder_size):  # Output logits.
        output_feed.append(self.outputs[l])

    # NOTE: added final arg for summaries
    # output_feed.append(merged_summaries)

    outputs = session.run(output_feed, input_feed)

    # NOTE: added outputs[-1] prefix with summaries to both exit paths
    if not forward_only:
      return outputs[-1], outputs[1], outputs[2], None  # Summaries, Gradient norm, loss, no outputs.
    else:
      return outputs[-1], None, outputs[0], outputs[1:]  # Summaries, No gradient norm, loss, outputs.

  def get_batch(self, data):
      encoder_size, decoder_size = self.max_sentence_length, self.max_sentence_length
      encoder_inputs, decoder_inputs = [], []

      # Get a random batch of encoder and decoder inputs from data,
      # pad them if needed, reverse encoder inputs and add GO to decoder.
      for _ in xrange(self.batch_size):
        encoder_input, decoder_input = random.choice(data)

        # Encoder inputs are padded and then reversed.
        encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
        encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

        # Decoder inputs get an extra "GO" symbol, and are padded then.
        decoder_pad_size = decoder_size - len(decoder_input) - 1
        decoder_inputs.append([data_utils.GO_ID] + decoder_input +
                              [data_utils.PAD_ID] * decoder_pad_size)

      # Now we create batch-major vectors from the data selected above.
      batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

      # Batch encoder inputs are just re-indexed encoder_inputs.
      for length_idx in xrange(encoder_size):
        batch_encoder_inputs.append(
            np.array([encoder_inputs[batch_idx][length_idx]
                      for batch_idx in xrange(self.batch_size)], dtype=np.int32))

      # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
      for length_idx in xrange(decoder_size):
        batch_decoder_inputs.append(
            np.array([decoder_inputs[batch_idx][length_idx]
                      for batch_idx in xrange(self.batch_size)], dtype=np.int32))

        # Create target_weights to be 0 for targets that are padding.
        batch_weight = np.ones(self.batch_size, dtype=np.float32)
        for batch_idx in xrange(self.batch_size):
          # We set weight to 0 if the corresponding target is a PAD symbol.
          # The corresponding target is decoder_input shifted by 1 forward.
          if length_idx < decoder_size - 1:
            target = decoder_inputs[batch_idx][length_idx + 1]
          if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
            batch_weight[batch_idx] = 0.0
        batch_weights.append(batch_weight)
      return batch_encoder_inputs, batch_decoder_inputs, batch_weights





