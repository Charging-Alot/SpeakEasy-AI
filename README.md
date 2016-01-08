#SpeakEasy-AI

SpeakEasy-AI is a machine learning project that aims to detect patterns in conversational responses.  You can talk to Marvin, our SpeakEasy chatbot, at <speakez.tk>.

##The Data##

Marvin is trained on data from Reddit comments.  The original dataset can be found [here](https://www.reddit.com/r/datasets/comments/3bxlg7/i_have_every_publicly_available_reddit_comment/), but since Reddit is insane corner of the internet, this data was heavily filtered to correct grammar and exclude long rants/comments containing uncommon words (Marvin's vocabulary is the 25,000 most common words in the entire dataset).  Because Reddit allows users to reply directly to another comment, comments are matched with their parent comment to form a prompt/response pair used for training.  If a comment has multiple children, the highest-voted one is used.   

##The Model##

Our model is an embedding Seq2Seq model built using Google's [Tensorflow](https://www.tensorflow.org/) library and is based initially on their [French-English translation model](http://arxiv.org/pdf/1506.05869.pdf).  It is made of LSTM cells, which have an internal cell state that changes as inputs (in our case, words in a sentence) are fed sequentially into the model.  This cell state allows the model to consider the context in which an input is recieved, and the output for a given input depends partially on the inputs that came before.  To learn more about LSTMs, check out [this blogpost](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) from Christopher Ola. Our model has 25,000 input and output nodes (one for each word in the vocabulary) and 3 hidden layers of 768 nodes each.   

##Try It Out##

If you have a conversational dataset you want to try out, or if you just want to tinker around, the code in this repo has been written so you can easily play around with model parameters and learning patterns to design your own version of a SpeakEasy chatbot. 

###Getting Started###

After cloning the repo, run `$ scripts/install.sh` from the root directory to configure Tensorflow and set up the virual env.  During this process, you will be asked if you want to configure Tensorflow for GPU.  Enabling GPU allows for significantly faster training but also requires CUDA/cuDUNN (you have to register a developer to download cuDUNN) and a NVIDIA graphics card.

###Training Your Model###

Use `$ scripts/run.sh` to train your model.  Running this program without --decode reads reads training and validation data files from the directory specified as --data_dir, builds out a vocabulary based on --vocab size, and tokenizes data by line.  The vocabulary and tokenized data are saved to disk after they are built, so data parsing only needs to occur once for a given vocabulary size (see data_utils.py for more information).  Training data is used to train a speakEasy model, and validation data is used to evaluate the perplexity of the model at each time-step.  Checkpoints are saved to the --train_dir directory every --steps_per_checkpoint, and event logs are saved to --log_dir and can be visualized using Tensorboard.     
Running with --decode starts an interactive loop that allows you to interact with the chatbot based on the most recent checkpoint saved in --train_dir.  Please note that to decode or to resume training from a previous checkpoint, the parameters describing the architecture of the model must be identical to the ones previously specified during training.  Also, it appears that you can only decode using CPU-only tensorflow (I'm not sure why, but running it with GPU gives a memory error) even if the model was previously trained on a GPU-enabled machine. 

Several model parameters can be customized using flags.  All flag arguments are optional since reasonable default values are provided in runtime_vaiables.py (except --data_dir, whose default needs to be overwritten to point to a data-containing directory):

Training parmeters:
|Flag|Description|
|:---:|:---:|
|--learning_rate LEARNING_RATE                          |Learning rate.                         |
|--learning_rate_decay_factor LEARNING_RATE_DECAY_FACTOR|Learning rate decays by this much.     |
|--max_gradient_norm MAX_GRADIENT_NORM                  |Clip gradients to this norm.           |
|--steps_per_checkpoint STEPS_PER_CHECKPOINT      |How many training steps to do per checkpoint.|

Model architecture:
|Flag|Description|
|:---:|:---:|
|--batch_size BATCH_SIZE                                |Batch size to use during training.     |
|--size SIZE                                            |Size of each model layer.              |
|--num_layers NUM_LAYERS                                |Number of layers in the model.         |
|--vocab_size VOCAB_SIZE                                |Vocabulary size.                       |
|--model_type MODEL_TYPE               |Seq2Seq model type: 'embedding_attention' or 'embedding'|
|--buckets BUCKETS                                      |Implement the model with buckets       |
|--nobuckets                                            |
|--max_sentence_length  MAX_SENTENCE_LENGTH   |Maximum sentence length for model WITHOUT buckets|

Data parameters:
|Flag|Description|
|:---:|:---:|
|--max_train_data_size MAX_TRAIN_DATA_SIZE    |Limit on the size of training data (0: no limit).|
  
Directories:
|Flag|Description|
|:---:|:---:|
|--data_dir DATA_DIR                                    |Data directory.                        |
|--train_dir TRAIN_DIR                                  |Training directory.                    |
|--log_dir LOG_DIR                                      |Logging directory.                     |
  
Testing:
|Flag|Description|
|:---:|:---:|
|--decode DECODE                                        |Set to True for interactive decoding.  |
|--nodecode                                             |                                       |
|--self_test SELF_TEST                                  |Set to True to run a self-test.        |
|--restore_model                                        |Path to model to restore.              |
