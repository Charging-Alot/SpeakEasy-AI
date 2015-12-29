"""Binary for training and testing SpeakEasy chatbot.

Bash script to run: scripts/run.sh

Running this program without --decode reads reads training and validation data files from the directory specified as --data_dir, builds out a vocabulary based on --vocab size, and tokenizes data by line.  The vocabulary and tokenized data are saved to disk after they are built, so data parsing only needs to occur once for a given vocabulary size (see data_utils.py for more information).  Training data is used to train a speakEasy model, and validation data is used to evaluate the perplexity of the model at each time-step.  Checkpoints are saved to the --train_dir directory every --steps_per_checkpoint, and event logs are saved to --log_dir and can be visualized using Tensorboard.     
Running with --decode starts an interactive loop that allows you to interact with the chatbot based on the most recent checkpoint saved in --train_dir.  Please note that to decode or to resume training from a previous checkpoint, the parameters describing the architecture of the model must be identical to the ones previously specified during training.  Also, it appears that you can only decode using CPU-only tensorflow (I'm not sure why, but running it with GPU gives a memory error) even if the model was previously trained on a GPU-enabled machine. 

Several model parameters can be customized using flags.  All flag arguments are optional since reasonable default values are provided in runtime_vaiables.py (except --data_dir, whose default needs to be overwritten to point to a data-containing directory):

  Training parmeters:
  --learning_rate LEARNING_RATE                             Learning rate.
  --learning_rate_decay_factor LEARNING_RATE_DECAY_FACTOR   Learning rate decays by this much.
  --max_gradient_norm MAX_GRADIENT_NORM                     Clip gradients to this norm.
  --steps_per_checkpoint STEPS_PER_CHECKPOINT               How many training steps to do per checkpoint.

  Model architecture:
  --batch_size BATCH_SIZE                                   Batch size to use during training.
  --size SIZE                                               Size of each model layer.
  --num_layers NUM_LAYERS                                   Number of layers in the model.
  --vocab_size VOCAB_SIZE                                   Vocabulary size.
  --model_type MODEL_TYPE                                   Seq2Seq model type: 'embedding_attention' or 'embedding'
  --buckets BUCKETS                                         Implement the model with buckets
  --nobuckets
  --max_sentence_length  MAX_SENTENCE_LENGTH                Maximum sentence length for model WITHOUT buckets

  Data parameters:
  --max_train_data_size MAX_TRAIN_DATA_SIZE                 Limit on the size of training data (0: no limit).
  
  Directories:
  --data_dir DATA_DIR                                       Data directory.
  --train_dir TRAIN_DIR                                     Training directory.
  --log_dir LOG_DIR                                         Logging directory.
  
  Testing:
  --decode DECODE                                           Set to True for interactive decoding.
  --nodecode
  --self_test SELF_TEST                                     Set to True to run a self-test.
  --restore_model                                           Path to model to restore.
"""
## TO DO: UPDATE THIS ^^

from __future__ import absolute_import
from __future__ import print_function

import sys
import tensorflow as tf

from model.train import train
from test.self_test import self_test
from test.decode import decode
from slack import slack

from runtime_variables import params 
print("In speak_easy.py")

def main(_):
  # try:
  print('in main function call')
  sys.stdout.flush()
  if params.self_test:
    self_test()
  elif params.decode:
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
