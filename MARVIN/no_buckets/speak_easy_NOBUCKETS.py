"""Binary for training chat model and interacting with it.

Running this program (python_model/scripts/run.sh) without --decode reads data and training files in the directory specified as --data_dir, tokenizes them in a very basic way, and then starts training a model.  Checkpoints are saved every --steps_per_checkpoint to --train_dir and event logging files are saved to --logdir.

Running with --decode starts an interactive loop so you can see how
the current checkpoint responds to user input.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time

import tensorflow.python.platform

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import data_utils
import seq2seq_NOBUCKETS as seq2seq_model
from tensorflow.python.platform import gfile


tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 256, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("vocab_size", 3000, "Vocabulary size.")
tf.app.flags.DEFINE_integer("max_sentence_length", 50, "Max sentence length.")
tf.app.flags.DEFINE_string("data_dir", "/tmp", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "/tmp", "Training directory.")
tf.app.flags.DEFINE_string("log_dir", "/tmp", "Log directory.")

tf.app.flags.DEFINE_string("model_type", "emedding_attention", "Seq2Seq model type") # NOTE: can be 'embedding_attention' or 'embedding'

tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False, ## Is this a concern?
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")

FLAGS = tf.app.flags.FLAGS

def read_data(data_path, max_size=None):
  """Read data from training file and compiles into dataset.

  Args:
    data_path: path to the files with token-ids
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).

  Returns:
    data_set: A list of (source, target) pairs read from the provided data files
  """
  data_set = []
  with gfile.GFile(data_path, mode="r") as data_file:    
    source, _ = data_file.readline(), data_file.readline()
    # target line is line after source line
    target = data_file.readline()
    counter = 0
    while source and target and (not max_size or counter < max_size):
      counter += 1
      if counter % 10000 == 0:
        print("  reading data line %d" % counter)
        sys.stdout.flush()
      source_ids = [int(x) for x in source.split()]
      target_ids = [int(x) for x in target.split()]
      target_ids.append(data_utils.EOS_ID)
      if len(source_ids) <= FLAGS.max_sentence_length and len(target_ids) <= FLAGS.max_sentence_length:
        data_set.append([source_ids, target_ids])
      source = target
      target = data_file.readline()
  return data_set


def create_model(session, forward_only):
  """Create SpeakEasy model and initialize or load parameters in session."""
  model = seq2seq_model.Seq2SeqModel(FLAGS.vocab_size, FLAGS.max_sentence_length, FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size, FLAGS.learning_rate, FLAGS.learning_rate_decay_factor,forward_only=forward_only, FLAGS.model_type)
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)

  if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    # Right now this is only good for getting the graph shape:
    writer = tf.train.SummaryWriter(FLAGS.log_dir, session.graph_def)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")

    ##TODO : write out stuff for logging history here
    # test_hist1 = tf.histogram_summary("name", tf.trainable_variables()[0])
    merged_summaries = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter(FLAGS.log_dir, session.graph_def)
    # tf.train.write_graph(session.graph_def, FLAGS.log_dir, 'graph.pbtxt')
    session.run(tf.initialize_all_variables())


  return model


def train():
  """Run SpeakEasy/server/python_model/scripts/run.sh to train model"""
  # prepare movie subtitle data.
  print("Preparing data in %s" % FLAGS.data_dir)
  sys.stdout.flush()
  data_train, data_dev, _ = data_utils.prepare_data(FLAGS.data_dir, FLAGS.vocab_size)

  with tf.Session() as sess:
    # create model.
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    sys.stdout.flush()
    model = create_model(sess, False)

    # set up event logging. NOTE: added this
    merged_summaries = tf.merge_all_summaries()
    # writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph_def)

    print ("Reading development and training data (limit: %d)."
           % FLAGS.max_train_data_size)
    sys.stdout.flush()
    dev_set = read_data(data_dev)
    train_set = read_data(data_train, FLAGS.max_train_data_size)

    # this is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    while True:

      # get a batch and make a step.
      start_time = time.time()
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          train_set)
      summaries, _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                   target_weights, False)
      step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
      loss += step_loss / FLAGS.steps_per_checkpoint
      current_step += 1

      # once in a while, we save checkpoint, print statistics, and run evals.
      if current_step % FLAGS.steps_per_checkpoint == 0:

        # print statistics for the previous epoch.
        perplexity = math.exp(loss) if loss < 300 else float('inf')
        print ("global step %d learning rate %.4f step-time %.2f perplexity "
               "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                         step_time, perplexity))
        sys.stdout.flush()
        # decrease learning rate if no improvement was seen over last 3 times.
        if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
          result = sess.run([model.learning_rate_decay_op])
        previous_losses.append(loss)
        # save checkpoint and zero timer and loss.
        checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        step_time, loss = 0.0, 0.0

def decode():
  """Run SpeakEasy/server/python_model/scripts/run.sh --decode for to chat with model"""
  with tf.Session() as sess:
    # create model and load parameters.
    model = create_model(sess, True)
    model.batch_size = 1  # decode one sentence at a time.

    # load vocabularies.
    vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab%d" % FLAGS.vocab_size)
    en_vocab, _ = data_utils.initialize_vocabulary(vocab_path)
    _, rev_fr_vocab = data_utils.initialize_vocabulary(vocab_path)

    # Decode from standard input.
    sys.stdout.write("> ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    while sentence:
      # Get token-ids for the input sentence.
      token_ids = data_utils.sentence_to_token_ids(sentence, en_vocab)
      # Which bucket does it belong to?
      bucket_id = min([b for b in xrange(len(_buckets))
                       if _buckets[b][0] > len(token_ids)])
      # Get a 1-element batch to feed the sentence to the model.
      encoder_inputs, decoder_inputs, target_weights = model.get_batch([(token_ids, [])])
      # Get output logits for the sentence.
      _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, True)
      # This is a greedy decoder - outputs are just argmaxes of output_logits.
      outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
      # If there is an EOS symbol in outputs, cut them at that point.
      if data_utils.EOS_ID in outputs:
        outputs = outputs[:outputs.index(data_utils.EOS_ID)]
      # Print out French sentence corresponding to outputs.
      print(" ".join([rev_fr_vocab[output] for output in outputs]))
      print("> ", end="")
      sys.stdout.flush()
      sentence = sys.stdin.readline()


def self_test():
  """Test the translation model."""
  with tf.Session() as sess:
    print("Self-test for neural translation model.")
    # Create model with vocabularies of 10, max sentence size of 10, 2 layers of 32.
    model = seq2seq_model.Seq2SeqModel(10, 10, 10, 32, 2,
                                       5.0, 32, 0.3, 0.99, num_samples=8, FLAGS.model_type)
    sess.run(tf.initialize_all_variables())

    # Fake data set
    data_set = ([([1, 1], [2, 2]), ([3, 3], [4]), ([5], [6]), ([1, 1, 1, 1, 1], [2, 2, 2, 2, 2]), ([3, 3, 3], [5, 6])])
    for _ in xrange(5):  # Train the fake model for 5 steps.
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          data_set)
      model.step(sess, encoder_inputs, decoder_inputs, target_weights, False)


def main(_):
  if FLAGS.self_test:
    self_test()
  elif FLAGS.decode:
    decode()
  else:
    train()

if __name__ == "__main__":
  tf.app.run()
