#!/bin/bash


declare -a arr=("A0201" "A0101" "A0301" "A0203" "A1101" "A0206" "A3101")
for ALLELE in "${arr[@]}"
do
    for LEARNING_RATE in 0.01 0.001 0.0001
    do
        for BATCH_SIZE in 60 70 80 100
        do
           python mhcPreds_tflearn_cmd_line.py -cmd  "train_test_eval" -b $BATCH_SIZE -e 30 -bn 1 -d 32 -c 'kmer_embedding' -a $ALLELE -m 'embedding_rnn' -r $LEARNING_RATE --run-id $ALLELE'_''kmer_embedding_batch_size'$BATCH_SIZE'_LR_'$LEARNING_RATE
        done
    done
done