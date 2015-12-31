#!/bin/bash 
COUNTER=3
while [  $COUNTER -lt 5941886 ]; do
 echo PROCESSING LINE $COUNTER
 venv/bin/python distributed/embedding.py --data_dir="/Volumes/HD/SPEAKEASY_DATA/REDDIT/reddit_data/data_25000_reddit" --readline=$COUNTER
 let COUNTER=COUNTER+2 
done
