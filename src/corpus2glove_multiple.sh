#!/bin/bash -e

OUT_DIR=data/working

WINDOW_SIZE=${1:-10}
MODEL=${2:-2} # 1:W, 2:W+C
VOCAB_MINCOUNT=100 # words with lower frequency are removed before windows
DISTANCE_WEIGHTING=1 # normalized co-occurrence counts (vanilla GloVe)
VECTOR_SIZE=300
ETA=0.05 # learning rate
MAX_ITER=100
SEED=1

files=$( ls corpora/wiki2021*.txt )

for corpus in ${files[@]}; do

    echo "Training vectors for $corpus..."
    src/corpus2glove.sh $corpus $OUT_DIR $VOCAB_MINCOUNT \
        $WINDOW_SIZE $DISTANCE_WEIGHTING $VECTOR_SIZE $ETA $MAX_ITER $MODEL $SEED

done
