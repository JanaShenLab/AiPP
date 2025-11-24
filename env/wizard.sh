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
wget https://zenodo.org/records/17693548/files/ligcysS_v1.tar.xz
echo "Extracting LigCys-S weights: this could take a few minutes"
tar -xJf ligcysS_v1.tar.xz

#ligcysA
echo "Fetching LigCys-A weights: this could take a few minutes"
wget https://zenodo.org/records/17693548/files/ligcysA_v1.tar.xz
echo "Extracting LigCys-A weights: this could take a few minutes"
tar -xJf ligcysA_v1.tar.xz

#ligbind
echo "Fetching LigBind weights: this could take a few minutes"
wget https://zenodo.org/records/17693713/files/ligbind_v1.tar.xz
echo "Extracting LigBind weights: this could take a few minutes"
tar -xJf ligbind_v1.tar.xz

#ssbind
echo "Fetching SSBind weights: this could take a few minutes"
wget https://zenodo.org/records/17693474/files/ssbind_v1.tar.xz
echo "Extracting SSBind weights: this could take a few minutes"
tar -xJf ssbind_v1.tar.xz

#znbind
echo "Fetching ZNBind weights: this could take a few minutes"
wget https://zenodo.org/records/17692131/files/znbind_v1.tar.xz
echo "Extracting ZNBind weights: this could take a few minutes"
tar -xJf znbind_v1.tar.xz

#housekeeping
echo "Peforming housekeeping"
rm ligcysS_v1.tar.xz
rm ligcysA_v1.tar.xz
rm ligbind_v1.tar.xz
rm ssbind_v1.tar.xz
rm znbind_v1.tar.xz
