#!/usr/local/bin/bash
# Runs the speak_easy model.

TRAIN_DIR='/Volumes/Seagate\ Backup Plus Drive/TrainingDirectories'
DATA_DIR='/Volumes/Seagate Backup Plus Drive/SPEAKEASY_DATA'
ROBOT_NAME=MARVIN



venv/bin/python speak_easy.py --train_dir='/Volumes/HD/SPEAKEASY_DATA/REDDIT/reddit_data/' --data_dir='/Volumes/HD/SPEAKEASY_DATA/REDDIT/reddit_data/data_25000_reddit' $@
