#!/bin/bash

#installation wizard 

#create python venv
python3 -m venv AiPP

#activate it
source AiPP/bin/activate

#update pip
pip install -U pip

#install reqs
pip install numpy torch esm tqdm httpx colorama

#download wts from zenodo

