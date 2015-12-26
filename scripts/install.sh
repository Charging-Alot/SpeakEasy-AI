#!/usr/bin/env bash
# Sets up the Python operating environment.

TENSORFLOW_PACKAGE=""

if [[ "$OSTYPE" == "linux-gnu" ]]; then
  echo "Do you want GPU support? [y/n]"
  read use_gpu

  echo "Configuring for Linux..."
  TENSORFLOW_PACKAGE=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.5.0-cp27-none-linux_x86_64.whl

  if [[ "$use_gpu" == "y" || "$use_gpu" == "Y" ]]; then
    echo "Using GPU!!!"
    TENSORFLOW_PACKAGE=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.5.0-cp27-none-linux_x86_64.whl
  fi

elif [[ "$OSTYPE" == "darwin"* ]]; then
  echo "Configuring for Mac OS..."
  TENSORFLOW_PACKAGE=https://storage.googleapis.com/tensorflow/mac/tensorflow-0.5.0-py2-none-any.whl
fi

if [ -d "venv" ]; then
  echo "Removing existing virtualenv..."
  rm -rf venv
fi

echo "Creating virtualenv..."

virtualenv venv

echo "Installing packages..."

# Tensorflow needs some special love
venv/bin/pip install --upgrade six
venv/bin/pip install --upgrade https://storage.googleapis.com/tensorflow/mac/tensorflow-0.5.0-py2-none-any.whl

venv/bin/pip install -r requirements.txt

echo "Done!"
