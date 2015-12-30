"""ChatBot for interactive conversations"""
import sys
import os
path = os.path.join(os.path.dirname(__file__), '..') 
sys.path.append(path)

import tensorflow as tf
import numpy as np

from data import data_utils
from model import model_utils
from runtime_variables import params

class ChatBot(object):
  def __init__(self):
    """Create the chatbot
    Initializes a tensorflow session, initialzes vocabulary and builds a model with a batch size of 1 for decoding 1 sentence at a time.
    """
    self.sess = tf.Session()
    vocab_path = os.path.join(params.data_dir, "vocab%d" % params.vocab_size)
    self.vocab, self.rev_vocab = data_utils.initialize_vocabulary(vocab_path)
    self.model = model_utils.create_model(self.sess, True)
    self.model.batch_size = 1 # Respond 1 sentence at a time.

  def respond(self, sentence):
    """Talk with the chatbot!
    Args:
      sentence: Sentence to be used as prompt for the bot.  Assumes that sentence has already been parsed/cleaned (see parse.js and parse.py).
    Return:
      A string that represents the bot's response.  
    """
        # Get token-ids for the input sentence.
    token_ids = data_utils.sentence_to_token_ids(sentence, self.vocab)
    try:
      # Which bucket does it belong to?
      bucket_id = min([b for b in xrange(len(self.model.buckets)) if self.model.buckets[b][0] > len(token_ids)])
      # Get a 1-element batch to feed the sentence to the model.
      encoder_inputs, decoder_inputs, target_weights = self.model.get_batch({bucket_id: [(token_ids, [])]}, bucket_id=bucket_id)
      # Get output logits for the sentence.
      _, _, _, output_logits = self.model.step(self.sess, encoder_inputs, decoder_inputs, target_weights, True, bucket_id=bucket_id)
    except AttributeError:
      encoder_inputs, decoder_inputs, target_weights = self.model.get_batch([(token_ids, [])], bucket_id=None)
      _, _, _, output_logits = self.model.step(self.sess, encoder_inputs, decoder_inputs, target_weights, True, bucket_id=None)
    # This is a greedy decoder - outputs are argmaxes of output_logits.
    outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
    # If there is an EOS symbol in outputs, cut them at that point.
    if data_utils.EOS_ID in outputs:
      outputs = outputs[:outputs.index(data_utils.EOS_ID)]
    # Return sentence corresponding to outputs.
    return " ".join([self.rev_vocab[output] for output in outputs])
