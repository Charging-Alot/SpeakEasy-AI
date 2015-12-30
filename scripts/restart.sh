#!/usr/local/bin/bash
#restarts server

kill -9 $(lsof -i:5000 -t) 2> /dev/null

venv/bin/python distributed/embedding.py --data_dir="/Volumes/HD/SPEAKEASY_DATA/REDDIT/reddit_data/data_25000_reddit"
