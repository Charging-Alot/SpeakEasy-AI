# import MARVIN.data_utils as data_utils
import data_utils
import numpy as np

class ChatBot(object):

  def __init__(self, vocab, rev_vocab, model, session):
    self.sess = session
    self.vocab = vocab
    self.rev_vocab = rev_vocab
    self.model = model
    self.model.batch_size = 1 # Respond 1 sentence at a time.

  def respond(self, sentence):
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
