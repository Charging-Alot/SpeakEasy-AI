from __future__ import print_function
from __future__ import absolute_import


import sys
import tensorflow as tf
from flask import Flask, jsonify, make_response, request, abort 

import speak_easy

app = Flask(__name__)


@app.route('/', methods=["POST"])
def generateResponse():
    # try:
    if not request.json or not 'prompt' in request.json:
      abort(400)
    response = Marvin.respond(request.json['prompt'])
    return make_response(jsonify({'response': response}), 200)
    # except NameError:
    #   return make_response(jsonify({'error': sys.exc_info()[0]}), 500)

if __name__ == '__main__':
    Marvin = speak_easy.initialize_chatbot()   
    app.run(debug=True)
