#!/bin/bash -e

CORPUS_ID=$1

SEEDS=(1)

corpus_in=corpora/$CORPUS_ID.txt

for seed in ${SEEDS[@]}; do

    corpus_out=corpora/${CORPUS_ID}s${seed}.txt
    echo "$corpus_out -- START"
    echo "Shuffling..."
    python3 -u src/data/shuffle_corpus.py $corpus_in $corpus_out $seed
    echo "Removing empty lines / useless whitespaces..."
    sed -i 's/^ *//; s/ *$//; /^$/d' $corpus_out
    echo "$corpus_out -- DONE"

done
