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
cd wts

#ligcysS
wget https://zenodo.org/records/17693548/files/ligcysS_v1.tar.xz
tar -xJf ligcysS_v1.tar.xz

#ligcysA
wget https://zenodo.org/records/17693548/files/ligcysA_v1.tar.xz
tar -xJf ligcysA_v1.tar.xz

#ligbind
wget https://zenodo.org/records/17693713/files/ligbind_v1.tar.xz
tar -xJf ligbind_v1.tar.xz

#ssbind
wget https://zenodo.org/records/17693474/files/ssbind_v1.tar.xz
tar -xJf ssbind_v1.tar.xz

#znbind
wget https://zenodo.org/records/17692131/files/znbind_v1.tar.xz
tar -xJf znbind_v1.tar.xz
