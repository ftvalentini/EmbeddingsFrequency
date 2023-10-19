#!/bin/bash

# GloVe + word2vec + PMI params
WINDOW=10 # window size
MINCOUNT=100 # vocab min count
# GloVe + word2vec params
SIZE=300 # vector dimension
SEED=1 # random seed
# word2vec + fasttext params
SG=1 # skipgram
NS=5 # negative sampling
NSE=0.75 # negative sampling exponent
###
# GloVe params
ETA=0.05 # initial learning rate
ITER=100 # max iterations
DIST_GLOVE=1 # distance weighting
# PMI params
DIST_COOC=0
CDS=0.75 # context distribution sampling (alpha)
NEG=5 # shift factor (k)

# context words
A=${1:-"SHE"}
B=${2:-"HE"}

files=("corpora/wiki2021.txt")
files+=("$( ls corpora/wiki2021_*_${B,,}*.txt )")

for f in ${files[@]}; do

    id=$( basename $f )
    id=${id%.txt}

    vocabfile="data/working/vocab-$id-V$MINCOUNT.txt"

    # SGNS:
    embedfile=data/working/w2v-$id-V$MINCOUNT-W$WINDOW-D$SIZE-SG$SG-S$SEED-NS$NS-NSE$NSE.npy
    outfile=results/bias_sgns-$id-$A-$B.csv
    python3 -u src/we2biasdf.py \
        --vocab $vocabfile --matrix $embedfile --a $A --b $B --out $outfile

    # FastText:
    embedfile=data/working/ft-$id-V$MINCOUNT-W$WINDOW-D$SIZE-SG$SG-S$SEED-NS$NS-NSE$NSE.npy
    outfile=results/bias_fasttext-$id-$A-$B.csv
    python3 -u src/we2biasdf.py \
        --vocab $vocabfile --matrix $embedfile --a $A --b $B --out $outfile

    # GloVe
    embedfile=data/working/glove-$id-V$MINCOUNT-W$WINDOW-D$DIST_GLOVE-D$SIZE-R$ETA-E$ITER-M2-S$SEED.npy
    outfile=results/bias_glove-$id-$A-$B.csv
    python3 -u src/we2biasdf.py \
        --vocab $vocabfile --matrix $embedfile --a $A --b $B --out $outfile

done
