from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf

import speak_easy

FLAGS = tf.app.flags.FLAGS

def test():
  Marvin = speak_easy.initialize_chatbot()
  results = []
  results.append("MODEL_TYPE: %s" % FLAGS.model_type)
  results.append("MODEL_SIZE: %d" % FLAGS.size)
  results.append("NUM_LAYERS: %d" % FLAGS.num_layers)
  results.append("VOCAB_SIZE: %d" % FLAGS.vocab_size)
  results.append("LEARNING_PATTERN: %d" % FLAGS.learning_rate_decay_factor)
  results.append("TRAINING DATA: %s" % FLAGS.training_data)
  results.append("CHECKPOINT: %s" % FLAGS.restore_model.split('-')[-1])

  with open('MARVIN/test_template', 'r') as template_file:
    for line in template_file:
      results.append(line)
      if "PROMPT" in line:
        sentence = line.split(':')[-1].strip()
        response = Marvin.respond(sentence)
        results.append("RESPONSE: %s\n" % response)
  # template_file.close() 

  with open('{0}/{1}'.format(FLAGS.test_dir, FLAGS.restore_model.split('-')[-1]), 'w') as test_file:
    test_file.write("".join(results))
  # test_file.close()


if __name__ == "__main__":
  test()
