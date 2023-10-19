#!/bin/bash

A=$1
B=$2

CORPUS="corpora/wiki2021.txt"
VOCAB="data/working/vocab-wiki2021-V100.txt"
SEED=33
OUTDIR="corpora"

echo "Counting frequency of $A and $B in each document"
python src/data/count_contexts_in_documents.py $CORPUS $A $B

echo "Resampling documents with $B"
python -u src/resample_corpora.py $CORPUS $VOCAB $A $B $OUTDIR $SEED

echo "Creating vocab of new corpora"
OUT_DIR=data/working
VOCAB_MINCOUNT=100
filenames=( $( ls corpora/wiki2021_undersampled_$B*.txt) ) 
filenames+=( $( ls corpora/wiki2021_oversampled_$B*.txt) ) 
for corpus in ${filenames[@]}; do
    src/corpus2vocab.sh $corpus $OUT_DIR $VOCAB_MINCOUNT
done







