from __future__ import print_function
from __future__ import absolute_import


import sys
# import tensorflow as tf
import boto3

from flask import Flask, jsonify, make_response, request, abort 

# import speak_easy

# s3 = boto3.resource('s3')
# Marvin = speak_easy.initialize_chatbot()   
application = Flask(__name__)


@application.route('/marvin', methods=["POST"])
def generateResponse():
    return make_response("WHEE", 200)
    # try:
    #   if not request.json or not 'prompt' in request.json:
    #     abort(400)
    #   response = Marvin.respond(request.json['prompt'])
    #   return make_response(jsonify({'response': response}), 200)
    # except:
    #   return make_response(jsonify({'error': sys.exc_info()[0]}), 500)

if __name__ == '__main__':
    application.run(debug=True)
