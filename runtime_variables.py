"""Model and logging parameters for Speaky chatbot

This program collects all the parameters for building, training and testing a chat model in a 'params' object.  Each of these parameters is either provided at runtime (as flags or environmental variables) or defaults its value listed below. 

"""
import tensorflow as tf
import os

#Only relevant if using bucketed model
buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]

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

tf.app.flags.DEFINE_integer("readline", 0, "Line to start reading for embedding.")

params = tf.app.flags.FLAGS
params.data_dir = os.environ.get('data_dir', params.data_dir)
params.restore_model = os.environ.get('restore_model', params.restore_model)

