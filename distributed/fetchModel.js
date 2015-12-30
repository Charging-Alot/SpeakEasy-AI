var AWS = require('aws-sdk');

var s3 = new AWS.S3();
var keys = require('./keys')

AWS.config.update({
  accessKeyId: keys.access,
  secretAccessKey: keys.secret,
  region: 'us-west-2'
});

module.exports = {};

var VARIABLES = [
  "projW",
  "projB",
  // "encoder_embedding",
  "LSTM_encoder1_weights",
  "LSTM_encoder1_biases",
  "LSTM_encoder2_weights",
  "LSTM_encoder2_biases",
  "LSTM_encoder3_weights",
  "LSTM_encoder3_biases",
  // "decoder_embedding",
  // "LSTM_decoder1_weights",
  // "LSTM_decoder1_biases",
  // "LSTM_decoder2_weights",
  // "LSTM_decoder2_biases",
  // "LSTM_decoder3_weights",
  // "LSTM_decoder3_biases",
];

var getVariableSync = function(variable) {
  console.log('fetching', variable)
  return new Promise(function(resolve, reject) {
    var params = {
      Bucket: 'speakeasy-data',
      Key: 'trainable_variables/' + variable
    };
    s3.getObject(params, function(err, data) {
      if (err) console.log(err);
      else {
        console.log('fetched', variable)
        var matrix = []
        rows = data.Body.toString().split("\n");
        rows.forEach(function(row) {
          matrix.push(row.split(' '));
        });
        console.log('parsed', variable);
        resolve(matrix);
      }
    });
  });
}

var fetchModel = function() {
  var variablePromises = [];
  VARIABLES.forEach(function(_, index) {
    variablePromises.push(getVariableSync(VARIABLES[index]));
  });
  Promise.all(variablePromises).then(function(fetchedVariables) {
    fetchedVariables.forEach(function(value, index) {
      module.exports[VARIABLES[index]] = value;
    });
    console.log('Variables loaded');
    console.log((Date.now() - start) / 1000)
    for (var key in module.exports) {
      console.log(key, typeof module.exports[key]);
    }
  });
}

// getVariableSync("LSTM_encoder3_weights")
var start = Date.now()
fetchModel();
