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
echo "Fetching LigCys-S weights: this could take a few minutes"
wget https://zenodo.org/records/19295949/files/ligcysS_v1.tar.gz
echo "Extracting LigCys-S weights: this could take a few minutes"
tar -xvzf ligcysS_v1.tar.gz

#ligcysA
echo "Fetching LigCys-A weights: this could take a few minutes"
wget https://zenodo.org/records/19295949/files/ligcysA_v1.tar.gz
echo "Extracting LigCys-A weights: this could take a few minutes"
tar -xvzf ligcysA_v1.tar.gz

#ligbind
echo "Fetching LigBind weights: this could take a few minutes"
wget https://zenodo.org/records/17693713/files/ligbind_v1.tar.xz
echo "Extracting LigBind weights: this could take a few minutes"
tar -xJf ligbind_v1.tar.xz

#ssbind!
echo "Fetching SSBind weights: this could take a few minutes"
wget https://zenodo.org/records/19052467/files/ssbind_v1.tar.gz
echo "Extracting SSBind weights: this could take a few minutes"
tar -xvzf ssbind_v1.tar.gz

#znbind!
echo "Fetching ZNBind weights: this could take a few minutes"
wget https://zenodo.org/records/19051390/files/znbind_v1.tar.gz
echo "Extracting ZNBind weights: this could take a few minutes"
tar -xvzf znbind_v1.tar.gz

#cubind!
echo "Fetching CUBind weights: this could take a few minutes"
wget https://zenodo.org/records/19051708/files/cubind_v1.tar.gz
echo "Extracting CUBind weights: this could take a few minutes"
tar -xvzf cubind_v1.tar.gz

#febind!
echo "Fetching FEBind weights: this could take a few minutes"
wget https://zenodo.org/records/19052041/files/febind_v1.tar.gz
echo "Extracting FEBind weights: this could take a few minutes"
tar -xvzf febind_v1.tar.gz

#fesbind --
echo "Fetching FeSBind weights: this could take a few minutes"
wget https://zenodo.org/records/19052104/files/fesbind_v1.tar.gz
echo "Extracting FeSBind weights: this could take a few minutes"
tar -xvzf fesbind_v1.tar.gz

#hembind --
echo "Fetching HEMBind weights: this could take a few minutes"
wget https://zenodo.org/records/19052343/files/hembind_v1.tar.gz
echo "Extracting HEMBind weights: this could take a few minutes"
tar -xvzf hembind_v1.tar.gz

#housekeeping
echo "Peforming housekeeping"
rm ligcysS_v1.tar.gz
rm ligcysA_v1.tar.gz
rm ligbind_v1.tar.xz
rm ssbind_v1.tar.gz
rm znbind_v1.tar.gz
rm cubind_v1.tar.gz
rm febind_v1.tar.gz
rm fesbind_v1.tar.gz
rm hembind_v1.tar.gz
