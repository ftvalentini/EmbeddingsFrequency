#!/bin/bash -e

VOCAB_MINCOUNT=100

WINDOW=${1:-10} # window size
NS=${2:-5}
NS_EXPONENT=${3:-0.75}
SIZE=300 # vector size
MINCOUNT=100
SG=1 # 0:cbow, 1:sgns
SEED=1

files=$( ls corpora/wiki2021*.txt )

for f in ${files[@]}; do

    id=$( basename $f )
    id=${id%.txt}
    vocabfile="data/working/vocab-$id-V$VOCAB_MINCOUNT.txt"

    echo "Training vectors for $f..."
    python3 -u src/corpus2fasttext.py --corpus $f --vocab $vocabfile \
        --size $SIZE --window $WINDOW --count $MINCOUNT --ns $NS \
        --ns_exponent $NS_EXPONENT --sg $SG --seed $SEED
    echo
done

