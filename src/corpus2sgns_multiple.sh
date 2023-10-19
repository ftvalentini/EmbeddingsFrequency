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

for corpus in ${files[@]}; do

    id=$( basename $corpus )
    id=${id%.txt}
    vocabfile="data/working/vocab-$id-V$VOCAB_MINCOUNT.txt"

    # if id is wiki2021 or wiki2021s1, then use --save_epochs flag
    if [[ $id == "wiki2021" || $id == "wiki2021s1" ]]; then
        save_epochs="--save_epochs"
    else
        save_epochs=""
    fi

    echo "Training vectors for $corpus..."
    python3 -u src/corpus2sgns.py --corpus $corpus --vocab $vocabfile \
        --size $SIZE --window $WINDOW --count $MINCOUNT --ns $NS \
        --ns_exponent $NS_EXPONENT --sg $SG --seed $SEED $save_epochs
    echo

done

