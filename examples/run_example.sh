#!/bin/bash

#activate the environment (assuming wizard was ran successfully)
source ../env/AiPP/bin/activate

#run AiPP on PTPN6
python ../aippCLI.py \
  --fasta ptpn6.fasta \
  --id ptpn6 \
  --out ptpn6.tsv
