#!/usr/local/bin/bash
# Updates current model
CURRENT_MODEL=$1

# Update current model 
aws s3 cp $CURRENT_MODEL s3://speak-easy/current_model/current.ckpt
eb deploy

# Update current model with new variables
venv/bin/python distributed/s3_saver.py --restore_model=$CURRENT_MODEL
