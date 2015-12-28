from __future__ import print_function
from __future__ import absolute_import

import sys
import os
import time
from threading import Thread

from flask import Flask, jsonify, make_response, request, abort 

from MARVIN import speak_easy

application = Flask(__name__)

Marvin = None
print("Loading SpeakEasy AI server")

def initialize():
    global Marvin
    print("Considering loading Marvin")

    if not Marvin:
        print("Going to load Marvin")
        Marvin = speak_easy.initialize_chatbot()
        print("Marvin is loaded")

@application.route('/', methods=["GET"])
def root():
    return make_response('fizzle bizzle %s' % time.time(), 200)

@application.route('/marvin', methods=["POST"])
def generate_response():
    if not Marvin:
      print("Marvin is not ready yet")
      abort(418)
      return

    try:
      if not request.json or not 'prompt' in request.json:
        abort(400)
      response = Marvin.respond(request.json['prompt'])
      return make_response(jsonify({'response': response}), 200)
    except:
      return make_response(jsonify({'error': sys.exc_info()[0]}), 500)

if __name__ == '__main__':
    initialize()
    application.run(debug=True)
else:
    Thread(target=initialize).start()
