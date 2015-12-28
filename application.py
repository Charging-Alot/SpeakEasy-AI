from __future__ import print_function
from __future__ import absolute_import

import sys
import os

from flask import Flask, jsonify, make_response, request, abort 

from MARVIN import speak_easy

Marvin = speak_easy.initialize_chatbot()   
application = Flask(__name__)

@application.route('/', methods=["GET"])
def root():
    return make_response('fizzle bizzle', 200)

@application.route('/marvin', methods=["POST"])
def generate_response():
    try:
      if not request.json or not 'prompt' in request.json:
        abort(400)
      response = Marvin.respond(request.json['prompt'])
      return make_response(jsonify({'response': response}), 200)
    except:
      return make_response(jsonify({'error': sys.exc_info()[0]}), 500)

if __name__ == '__main__':
    application.run(debug=False)
