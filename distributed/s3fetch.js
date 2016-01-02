var s3 = require('./awsConfig');
var trainableVariables = {};
var dataSet = [];

// These are the trainable variables stored on S3 
var VARIABLES = [
  "LSTM_encoder1_weights",
  "LSTM_encoder1_biases",
  "LSTM_encoder2_weights",
  "LSTM_encoder2_biases",
  "LSTM_encoder3_weights",
  "LSTM_encoder3_biases",
  "LSTM_decoder1_weights",
  "LSTM_decoder1_biases",
  "LSTM_decoder2_weights",
  "LSTM_decoder2_biases",
  "LSTM_decoder3_weights",
  "LSTM_decoder3_biases",
];


var s3Fetch = function(item, folder) {
  var key = folder ? folder + '/' + item : item
  console.log('fetching', key)
  return new Promise(function(resolve, reject) {
    var params = {
      Bucket: 'speakeasy-data',
      Key: key
    };
    s3.getObject(params, function(err, data) {
      if (err) console.log(err);
      else {
        console.log('fetched', item);
        resolve(JSON.parse(data.Body.toString()));
      }
    });
  });
}

// Fetches all trainable variables and stores them in trainableVariables object under their name 
// NOTE: this is a BEAR for your memory and takes 10 minutes
var fetchEntireModel = function() {
  var start = Date.now();
  var variablePromises = [];
  VARIABLES.forEach(function(_, index) {
    variablePromises.push(s3Fetch(VARIABLES[index], "trainable_variables"));
  });
  Promise.all(variablePromises).then(function(fetchedVariables) {
    fetchedVariables.forEach(function(value, index) {
      trainableVariables[VARIABLES[index]] = value;
    });
    console.log('Time to fetch model', Date.now() - start);
    console.log('Variables loaded');
  });
}

// Returns an object with "encoder" and "decoder" properties containing embedded input arrays
var fetchTrainingPair = function() {
  // Choose a random piece of data
  var randomIndex = Math.floor(Math.random() * dataSet.length);
  return s3Fetch(dataSet[randomIndex].Key);
}

// This can go away and become a global data length variable once all the data is uploaded and serialized (right now the files don't necessarily go in order beause the session fails on big inputs)
var countDataAvailable = function() {
  return new Promise(function(resolve, reject) {
    s3.listObjects({
      Bucket: 'speakeasy-data',
      Prefix: 'embedded/',
    }, function(err, data) {
      if (err) reject(err);
      else resolve(data.Contents);
    });
  });
}

///// Example of how to use this module:
countDataAvailable().then(function(data) {
  // Need to fetch dataSet first 
  dataSet = data;
  // Do whatever you want
  fetchTrainingPair();
});
