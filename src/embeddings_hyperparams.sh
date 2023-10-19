#!/bin/bash -e

# fixed params
VOCAB_MINCOUNT=100 # w2v, ft, glove
VECTOR_SIZE=300 # w2v, ft, glove
SEED=1 # w2v, ft, glove
SG=1 # w2v, ft
DISTANCE_WEIGHTING=1 # glove
ETA=0.05 # glove
MAX_ITER=100 # glove

# variable params
WINDOW_SIZE=(2 5 10) # w2v, ft, glove
NS=(1 5 15) # w2v, ft
NS_EXPONENT=(0.75 1) # w2v, ft
MODEL=(1 2) # glove (1:W, 2:W+C) -- sgns and ft always yield both


FILES=( corpora/wiki2021s1.txt )

for corpus in ${FILES[@]}; do

    id=$( basename $corpus )
    id=${id%.txt}
    vocabfile="data/working/vocab-$id-V$VOCAB_MINCOUNT.txt"

    for w in ${WINDOW_SIZE[@]}; do
        
        for ns in ${NS[@]}; do
            for nse in ${NS_EXPONENT[@]}; do

                    echo "Training W2V for $corpus with window=$w, ns=$ns, ns_exponent=$nse"
                    python3 -u src/corpus2sgns.py \
                        --corpus $corpus --vocab $vocabfile \
                        --size $VECTOR_SIZE --window $w --count $VOCAB_MINCOUNT --ns $ns \
                        --ns_exponent $nse --sg $SG --seed $SEED
                    echo

                    echo "Training FT for $corpus with window=$w, ns=$ns, ns_exponent=$nse"
                    python3 -u src/corpus2fasttext.py \
                        --corpus $corpus --vocab $vocabfile \
                        --size $VECTOR_SIZE --window $w --count $VOCAB_MINCOUNT --ns $ns \
                        --ns_exponent $nse --sg $SG --seed $SEED
                    echo

            done
        done

        for m in ${MODEL[@]}; do
            echo "Training GLOVE for $corpus with window=$w, model=$m"
            src/corpus2glove.sh $corpus $OUT_DIR $VOCAB_MINCOUNT \
                $w $DISTANCE_WEIGHTING $VECTOR_SIZE $ETA $MAX_ITER $m $SEED
            echo
        done    

    done

done

