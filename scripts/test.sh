#!/usr/local/bin/bash
# Runs the speak_easy model.

vocab_size=25000
checkpoint_dir='/Volumes/HD/SPEAKEASY_DATA/REDDIT/reddit_data'
data_dir='/Volumes/HD/SPEAKEASY_DATA/REDDIT/reddit_data/data_25000_reddit'

function run_test_cases() {
  for f in $(ls $1)
  do
    if ! [ $f == 'test' ] && ! [ -e $1/test/${f:20} ]; then
      echo "Processing $f file..."
      venv/bin/python test/test.py \
        --vocab_size=$vocab_size \
        --data_dir=$data_dir \
        --test_dir=$1/test \
        --restore_model=$1/$f \
        --model_type="embedding" \
        --size=$2 \
        --learning_rate_decay_factor=$3 \
        --training_data=$4 
    else
      echo "Already processed $f"
    fi
  done
}

run_test_cases "$checkpoint_dir/train_reddit_vocab25000_size256_dataHALF_learning5" 256 .5 'HALF'
run_test_cases "$checkpoint_dir/train_reddit_vocab25000_size768_dataHALF_learning5" 768 .5 'HALF'
run_test_cases "$checkpoint_dir/train_reddit_vocab25000_size768_dataFULL_learning5" 768 .5 'FULL'
run_test_cases "$checkpoint_dir/train_reddit_vocab25000_size768_dataFULL_learning99" 768 .99 'FULL'


