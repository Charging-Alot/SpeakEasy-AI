"""Binary for training and chatting with a SpeakEast chatbot.

Bash script to run: ../scripts/run.sh

Running this program without --decode reads reads training and validation data files (currently MASTER_FILE and VALIDATION, respectively) from the directory specified as --data_dir, builds out a vocabulary based on --vocab size, and tokenizes data by line.  The vocabulary and tokenized data are saved to disk after they are built, so data parsing only needs to occur once for a given vocabulary size (see data_utils.py for more information).  Training data is used to train a speakEasy model built according FLAGS-specified parameters, and validation data is used to evaluate the perplexity of the model at each time-step.  Checkpoints are saved to the --train_dir directory every --steps_per_checkpoint, and event logs are saved to --log_dir and can be visualized using Tensorboard.     
Running with --decode starts an interactive loop that allows you to interact with the chatbot based on the most recent checkpoint saved in --train_dir.  Please note that to decode or to resume training from a previous checkpoint, the FLAGS parameters describing the architecture of the model must be identical to the ones previously specified during training.  Also, it appears that you can only decode using CPU-only tensorflow (I'm not sure why, but running it with GPU gives a memory error). 

All flags arguments are optional since reasonable default values are provided below (except --data_dir, whose default needs to be overwritten to point to a data-containing directory):
  --learning_rate LEARNING_RATE                             Learning rate.
  --learning_rate_decay_factor LEARNING_RATE_DECAY_FACTOR   Learning rate decays by this much.
  --max_gradient_norm MAX_GRADIENT_NORM                     Clip gradients to this norm.
  --batch_size BATCH_SIZE                                   Batch size to use during training.
  --size SIZE                                               Size of each model layer.
  --num_layers NUM_LAYERS                                   Number of layers in the model.
  --vocab_size VOCAB_SIZE                                   Vocabulary size.
  --data_dir DATA_DIR                                       Data directory.
  --train_dir TRAIN_DIR                                     Training directory.
  --log_dir LOG_DIR                                         Logging directory.
  --model_type MODEL_TYPE                                   Seq2Seq model type: 'embedding_attention' or 'embedding'
  --max_train_data_size MAX_TRAIN_DATA_SIZE                 Limit on the size of training data (0: no limit).
  --steps_per_checkpoint STEPS_PER_CHECKPOINT               How many training steps to do per checkpoint.
  --decode DECODE                                           Set to True for interactive decoding.
  --nodecode
  --self_test SELF_TEST                                     Set to True to run a self-test.
  --buckets BUCKETS                                         Implement the model with buckets
  --nobuckets
  --max_sentence_length  MAX_SENTENCE_LENGTH                Maximum sentence length for model WITHOUT buckets
"""
## TO DO: UPDATE THIS ^^

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time

import numpy as np
from six.moves import xrange 
import tensorflow as tf

import tensorflow.python.platform
from tensorflow.python.platform import gfile

import MARVIN.data_utils as data_utils
import MARVIN.seq2seq_model as seq2seq_model
import MARVIN.chat_bot as chat_bot
import MARVIN.slack as slack


# Training params
tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.5, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200, "How many training steps to do per checkpoint.")

# Model arcitecture
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 768, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("vocab_size", 25000, "Vocabulary size.")
tf.app.flags.DEFINE_string("model_type", "embedding", "Seq2Seq model type: 'embedding_attention' or 'embedding'")
tf.app.flags.DEFINE_boolean("buckets", True, "Implement the model with buckets")
tf.app.flags.DEFINE_integer("max_sentence_length", 50, "Maximum sentence length for model WITHOUT buckets")

# Data params
tf.app.flags.DEFINE_integer("max_train_data_size", 0, "Limit on the size of training data (0: no limit).")

# Directories
tf.app.flags.DEFINE_string("data_dir", "/tmp", "Data directory.")
tf.app.flags.DEFINE_string("train_dir", "/tmp", "Training directory.")
tf.app.flags.DEFINE_string("log_dir", "/tmp", "Logging directory.")
tf.app.flags.DEFINE_string("test_dir", "/tmp", "Testing directory.")

# Testing params
tf.app.flags.DEFINE_boolean("decode", False, "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False, "Set to True to run a self-test.")
tf.app.flags.DEFINE_string("restore_model", "", "Path to model to restore.")
tf.app.flags.DEFINE_string("training_data", "FULL", "Data set used to train model (for logging in test files).")

# tf.app.flags.DEFINE_string("learning_pattern", ".99", "Learning pattern used to train model.")

FLAGS = tf.app.flags.FLAGS

data_dir = os.environ.get('data_dir', FLAGS.data_dir)
restore_model = os.environ.get('restore_model', FLAGS.restore_model)

# Only relevant if we are using a bucketed model
_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]

print(FLAGS.max_sentence_length, "sentence length??")

def read_data(data_path, max_size=None):
  """Read data from data files to compile data set.  Each line is used twice, once as a prompt and once as a response.

  Args:
    data_path: path to the file with token-ids
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).

  Returns:
    if using a model with buckets:
      data_set: a list of length len(_buckets); data_set[n] contains a list of
        (prompt, response) pairs read from the provided data files that fit
        into the n-th bucket, i.e., such that len(prompt) < _buckets[n][0] and
        len(response) < _buckets[n][1]; prompt and response are lists of token-ids.
    else:
      data_set: a list of (prompt, response) pairs read from the provided data files;
        prompt and response are lists of token-ids.
  """
  data_set = [[] for _ in _buckets] if FLAGS.buckets else []
  with gfile.GFile(data_path, mode="r") as data_file:    
    # Response line is line after prompt line
    prompt, response = data_file.readline(), data_file.readline()
    counter = 0
    while prompt and response and (not max_size or counter < max_size):
      # Skip empty lines
      if (len(prompt.strip().split()) > 1) and (len(response.strip().split()) > 1):
        counter += 1
        if counter % 10000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()
        prompt_ids = [int(x) for x in prompt.split()]
        response_ids = [int(x) for x in response.split()]
        response_ids.append(data_utils.EOS_ID)
        if FLAGS.buckets:
          for bucket_id, (prompt_size, response_size) in enumerate(_buckets):
            if len(prompt_ids) < prompt_size and len(response_ids) < response_size:
              data_set[bucket_id].append([prompt_ids, response_ids])
              break
        else:
          if len(prompt_ids) <= FLAGS.max_sentence_length and len(response_ids) <= FLAGS.max_sentence_length:
            data_set.append([prompt_ids, response_ids])
      prompt, response = data_file.readline(), data_file.readline()
    epoch = counter / FLAGS.batch_size
  return data_set, epoch


def create_model(session, forward_only):
  """Create speakEasy model and initialize or load parameters in session."""
  if FLAGS.buckets:
    model = seq2seq_model.Seq2SeqModel(
        FLAGS.vocab_size, _buckets,
        FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
        FLAGS.learning_rate, FLAGS.learning_rate_decay_factor, FLAGS.model_type,
        forward_only=forward_only)
  else:
    model = seq2seq_model.Seq2SeqModel(
        FLAGS.vocab_size, FLAGS.max_sentence_length, 
        FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size, 
        FLAGS.learning_rate, FLAGS.learning_rate_decay_factor, FLAGS.model_type, 
        forward_only=forward_only)
  
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)

  if restore_model:
    print("Reading model parameters from %s" % restore_model)
    model.saver.restore(session, restore_model)
  else:  
    if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
      print("Reading model parameters from %s" % ckpt)
      writer = tf.train.SummaryWriter(FLAGS.log_dir, session.graph_def)
      model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
      print("Created model with fresh parameters.")
      writer = tf.train.SummaryWriter(FLAGS.log_dir, session.graph_def)
      # tf.train.write_graph(session.graph_def, FLAGS.log_dir, 'graph.pbtxt')
      session.run(tf.initialize_all_variables())

  return model


def train():
  """Run SpeakEasy/server/python_model/scripts/run.sh to train model"""
  
  slack.connection.notify(
    text='Training SpeakEasy!',
  )
  # Prepare reddit data.
  print("Preparing data in %s" % data_dir)
  sys.stdout.flush()
  data_train, data_dev, _ = data_utils.prepare_data(data_dir, FLAGS.vocab_size)

  with tf.Session() as sess:
    # Create model.
    print("Creating %s model with %d layers of %d units." % (FLAGS.model_type, FLAGS.num_layers, FLAGS.size))
    sys.stdout.flush()
    if FLAGS.buckets: print("Using bucketed model.")
    sys.stdout.flush()
    model = create_model(sess, False)

    # Set up event logging. NOTE: added this, this is not finished
    merged_summaries = tf.merge_all_summaries()
    # writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph_def)

    # Read data into buckets and compute their sizes.
    print ("Reading development and training data (limit: %d)." % FLAGS.max_train_data_size)
    sys.stdout.flush()
    dev_set, _ = read_data(data_dev)
    train_set, epoch = read_data(data_train, FLAGS.max_train_data_size)

    if FLAGS.buckets:
      train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
      train_total_size = float(sum(train_bucket_sizes))
      # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
      # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
      # the size if i-th training bucket, as used later.
      train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size for i in xrange(len(train_bucket_sizes))]

    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    while True:
    # while current_step <= epoch * 7.5:
      # Get a batch and make a step.
      start_time = time.time()

      if FLAGS.buckets:
        # Choose a bucket according to data distribution. We pick a random number
        # in [0, 1] and use the corresponding interval in train_buckets_scale.
        random_number_01 = np.random.random_sample()
        bucket_id = min([i for i in xrange(len(train_buckets_scale)) if train_buckets_scale[i] > random_number_01])
        encoder_inputs, decoder_inputs, target_weights = model.get_batch(train_set, bucket_id=bucket_id)
        summaries, _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, False, bucket_id=bucket_id)
      else:
        encoder_inputs, decoder_inputs, target_weights = model.get_batch(train_set, bucket_id=None)
        summaries, _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, False, bucket_id=None)
      step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
      loss += step_loss / FLAGS.steps_per_checkpoint
      current_step += 1

      ## STUFF HERE ABOUT EPOCHS
      # if current_step % epoch == 0 and current_step >= epoch * 5:
      #   sess.run([model.learning_rate_decay_op])
      # Once in a while, we save checkpoint, print statistics, and run evals.
      if current_step % FLAGS.steps_per_checkpoint == 0:

        # Save summaries. NOTE: added this
        # result = sess.run(merged_summaries)
        # summary_str = result[0]
        # writer.add_summary(summary_str, current_step)

        # Print statistics for the previous epoch.
        perplexity = math.exp(loss) if loss < 300 else float('inf')
        log_line = ("global step %d learning rate %.4f step-time %.2f perplexity %.2f" % 
            (model.global_step.eval(), model.learning_rate.eval(),step_time, perplexity))
        print(log_line)
        sys.stdout.flush()
        slack.connection.notify(
          text=log_line,
        )

        # Decrease learning rate if no improvement was seen over last 3 times.
        if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
          result = sess.run([model.learning_rate_decay_op])
        previous_losses.append(loss)

        # Save checkpoint and zero timer and loss.
        checkpoint_path = os.path.join(FLAGS.train_dir, "speakEasy" + str(FLAGS.vocab_size) + ".ckpt")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        step_time, loss = 0.0, 0.0

        if FLAGS.buckets:
          # Run evals on development set and print their perplexity.
          for bucket_id in xrange(len(_buckets)-1):
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(dev_set, bucket_id)
            _, _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, True, bucket_id=bucket_id)
            eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
            log_line = "eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx)
            print("  %s" % log_line)
            slack.connection.notify(
              text=log_line,
            )
            sys.stdout.flush()

def initialize_chatbot():
  """Build interactive chatbot"""
  sess = tf.Session()
  model = create_model(sess, True)
  vocab_path = os.path.join(data_dir, "vocab%d" % FLAGS.vocab_size)
  vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)
  return chat_bot.ChatBot(vocab, rev_vocab, model, sess)
  
def decode():
  """Interactive decoding of restored model"""
  Marvin = initialize_chatbot()
  sys.stdout.write("> ")
  sys.stdout.flush()
  sentence = sys.stdin.readline()
  while sentence:
    response = Marvin.respond(sentence)
    print(response)
    print("> ", end="")
    sys.stdout.flush()
    sentence = sys.stdin.readline()


def self_test():
  """Test the training protocol."""
  with tf.Session() as sess:
    print("Self-test for neural translation model.")
    # Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
    model = seq2seq_model.Seq2SeqModel(10, 10, [(3, 3), (6, 6)], 32, 2, 5.0, 32, 0.3, 0.99, num_samples=8)
    sess.run(tf.initialize_all_variables())
    # Fake data set for both the (3, 3) and (6, 6) bucket.
    data_set = ([([1, 1], [2, 2]), ([3, 3], [4]), ([5], [6])], [([1, 1, 1, 1, 1], [2, 2, 2, 2, 2]), ([3, 3, 3], [5, 6])])
    for _ in xrange(5):  # Train the fake model for 5 steps.
      if FLAGS.buckets:
        bucket_id = random.choice([0, 1])
        encoder_inputs, decoder_inputs, target_weights = model.get_batch(data_set, bucket_id=bucket_id)
        model.step(sess, encoder_inputs, decoder_inputs, target_weights, True, bucket_id=bucket_id)
      else:
        encoder_inputs, decoder_inputs, target_weights = model.get_batch(data_set, bucket_id=None)
        model.step(sess, encoder_inputs, decoder_inputs, target_weights, True, bucket_id=None)


def main(_):
  # try:
  print('in main function call')
  sys.stdout.flush()
  if FLAGS.self_test:
    self_test()
  elif FLAGS.decode:
    decode()
  else:
    print ('training')
    train()
  # except Exception as e:
  #   slack.connection.notify(
  #     text='SpeakEasy shutting down!',
  #     fields=[{
  #       'title': 'Error',
  #       'value': str(e),
  #     }],
  #   )
  #   raise e

if __name__ == "__main__":
  print('Running app')
  tf.app.run()
