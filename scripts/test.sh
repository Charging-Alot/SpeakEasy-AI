#!/usr/local/bin/bash
# Runs the speak_easy model.

vocab_size=25000
data_dir='/Volumes/HD/SPEAKEASY_DATA/REDDIT/reddit_data/data_25000_reddit'
echo 'hi'
FILES='/Volumes/HD/SPEAKEASY_DATA/REDDIT/reddit_data/train_reddit_vocab25000_size256_dataHALF_learning5'
for f in $FILES
do
  echo "Processing $f file..."
  venv/bin/python test/test.py --vocab_size=$vocab_size --data_dir=$data_dir --test_dir=$FILES --restore_model=$f --model_type="embedding" --size=256 --learning_rate_decay_factor=.5 --training_data='HALF' 
done

FILES='/Volumes/HD/SPEAKEASY_DATA/REDDIT/reddit_data/train_reddit_vocab25000_size768_dataHALF_learning5'
for f in $FILES
do
  echo "Processing $f file..."
  venv/bin/python test/test.py --vocab_size=$vocab_size --data_dir=$data_dir --test_dir=$FILES --restore_model=$f --model_type="embedding" --size=768 --learning_rate_decay_factor=.5 --training_data='HALF' 
done

FILES='/Volumes/HD/SPEAKEASY_DATA/REDDIT/reddit_data/train_reddit_vocab25000_size768_dataFULL_learning5'
for f in $FILES
do
  echo "Processing $f file..."
  venv/bin/python test/test.py --vocab_size=$vocab_size --data_dir=$data_dir --test_dir=$FILES --restore_model=$f --model_type="embedding" --size=768 --learning_rate_decay_factor=.5 --training_data='FULL' 
done

FILES='/Volumes/HD/SPEAKEASY_DATA/REDDIT/reddit_data/train_reddit_vocab25000_size768_dataFULL_learning99'
for f in $FILES
do
  echo "Processing $f file..."
  venv/bin/python test/test.py --vocab_size=$vocab_size --data_dir=$data_dir --test_dir=$FILES --restore_model=$f --model_type="embedding" --size=768 --learning_rate_decay_factor=.9  --training_data='FULL' 
done

