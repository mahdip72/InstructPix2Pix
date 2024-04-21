#!/bin/bash

pip install uv
conda install --yes pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install --yes xformers -c xformers
conda install --yes h5py
conda install --yes matplotlib
conda install --yes pandas
uv pip install accelerate
uv pip install transformers
uv pip install diffusers
uv pip install datasets
uv pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'
uv pip install tensorboard
uv pip install python-box
uv pip install peft
uv pip install -U scikit-learn
