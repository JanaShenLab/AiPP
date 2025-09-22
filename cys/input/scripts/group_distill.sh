#!/bin/bash

timestamp=$(date +%b%d | tr [:upper:] [:lower:])

gt=0
scb=1

ngt=${gt}
nscb=${scb}

let cb=${gt}+1
let ncb=${ngt}+1

#sanity check
if [ ${scb} -gt ${cb} ]; then
  echo "scb cannot be less than cb, it's nonsensical.."
  exit;
fi

uniprotDB=/home/wayyne/lab/uniprot/uniprot_sprot_simple.fasta 

LARGE_cache=aug27_s4r4-s4r4_PRO/dataset_cache.json
LARGE_cluster=aug27_s4r4-s4r4_PRO/cluster_cache.json

# #PRO version - fresh fetch/validation and cache creation
# #echo -e 'u\na\na\nu\n' | 
# aipp-group-distiller --dir from_master \
#   --uniprotdb ${uniprotDB} \
#   --cache ${LARGE_cache} \
#   --cluster-cache ${LARGE_cluster} \
#   --report-dir ${timestamp}_s${scb}r${cb}-s${nscb}r${ncb}_PRO \
#   --threads 60 \
#   --emb ../embeddings/esmCb6 \
#   --repr 76 \
#   --cb ">${gt}" \
#   --negcb ">${ngt}" \
#   --nostrictvalidation \
#   --add_unseen \
#   --unmaskroi C \
#   --min_src_per_pos ${scb} \
#   --min_src_per_neg ${nscb} \
#   --dedup-priority LS2024 \
#   --prioritize LS2024 \
#   --filterfor LS2024 \
#   --nmlb LS2024 \
#   --novote NMLB \
#   --repsonly \
#   --masknonreps
#  read -n 1 -s -r -p "press any key to continue..."

# #now a non-PRO version
# echo -e 'u\na\na\nu\n' | aipp-group-distiller --dir from_master \
#   --cluster-cache ${LARGE_cluster} \
#   --cache ${LARGE_cache} \
#   --uniprotdb ${uniprotDB} \
#   --report-dir ${timestamp}_s${scb}r${cb}-s${nscb}r${ncb}_NPR \
#   --threads 60 \
#   --emb ../embeddings/esmCb6 \
#   --repr 76 \
#   --cb ">${gt}" \
#   --negcb ">${ngt}" \
#   --nostrictvalidation \
#   --add_unseen \
#   --unmaskroi C \
#   --min_src_per_pos ${scb} \
#   --min_src_per_neg ${nscb} \
#   --dedup-priority LS2024 \
#   --prioritize LS2024 \
#   --filterfor LS2024 \
#   --nmlb LS2024 \
#   --novote NMLB 
#  read -n 1 -s -r -p "press any key to continue..."

 #finally a non-PRO, nopos_res, noneg_req version
 echo -e 'u\na\na\nu\n' | aipp-group-distiller --dir from_master \
   --cluster-cache ${LARGE_cluster} \
   --cache ${LARGE_cache} \
   --uniprotdb ${uniprotDB} \
   --report-dir ${timestamp}_s${scb}r${cb}-s${nscb}r${ncb}_ALL \
   --threads 60 \
   --emb ../embeddings/esmCb6 \
   --repr 76 \
   --cb ">${gt}" \
   --negcb ">${ngt}" \
   --nostrictvalidation \
   --add_unseen \
   --unmaskroi C \
   --min_src_per_pos ${scb} \
   --min_src_per_neg ${nscb} \
   --dedup-priority LS2024 \
   --prioritize LS2024 \
   --filterfor LS2024 \
   --nmlb LS2024 \
   --noposrequired --nonegrequired \
   --novote NMLB 

# echo "-----------------------------------------------------------------------"
# echo "All done! Enjoy."
#
## ABPP only data for manuscript text
#LARGE_cache=aug26_s4r4-s4r4_PRO/dataset_cache.json
#LARGE_cluster=aug26_s4r4-s4r4_PRO/cluster_cache.json
#uniprotDB=/home/wayyne/lab/uniprot/uniprot_sprot_simple.fasta 
#
#aipp-group-distiller --dir ABPPonly_from_master \
#  --uniprotdb ${uniprotDB} \
#  --report-dir ${timestamp}_s${scb}r${cb}-s${nscb}r${ncb}_PRO_ABPPonly \
#  --threads 60 \
#  --emb ../embeddings/esmCb6 \
#  --repr 76 \
#  --cb ">${gt}" \
#  --negcb ">${ngt}" \
#  --nostrictvalidation \
#  --add_unseen \
#  --unmaskroi C \
#  --min_src_per_pos ${scb} \
#  --min_src_per_neg ${nscb} \
#  --repsonly \
#  --masknonreps
# read -n 1 -s -r -p "press any key to continue..."
