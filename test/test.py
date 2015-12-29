"""Tests saved model on standard prompts

Bash script to run: scripts/test.sh

Running this program will initialize a SpeakEasy chatbot by building a fresh model and restoring the parameters from a previously saved checkpoint.  This chatbot will be tested using the prompts in 'test_template' and the results (along with identifying model information) will be written to a file named after the global step where the checkpoint was saved.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf

from runtime_variables import params

def test():
  Marvin = speak_easy.initialize_chatbot()
  results = []
  results.append("MODEL_TYPE: %s\n" % params.model_type)
  results.append("MODEL_SIZE: %d\n" % params.size)
  results.append("NUM_LAYERS: %d\n" % params.num_layers)
  results.append("VOCAB_SIZE: %d\n" % params.vocab_size)
  results.append("LEARNING_PATTERN: %d\n" % params.learning_rate_decay_factor)
  results.append("TRAINING DATA: %s\n" % params.training_data)
  results.append("CHECKPOINT: %s\n" % params.restore_model.split('-')[-1])

  with open('MARVIN/test_template', 'r') as template_file:
    for line in template_file:
      results.append(line)
      if "PROMPT" in line:
        sentence = line.split(':')[-1].strip()
        response = Marvin.respond(sentence)
        results.append("RESPONSE: %s\n" % response)
  # template_file.close() 

  with open('{0}/{1}'.format(params.test_dir, params.restore_model.split('-')[-1]), 'w') as test_file:
    test_file.write("".join(results))
  # test_file.close()


if __name__ == "__main__":
  test()
