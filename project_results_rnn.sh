#!/bin/bash


declare -a arr=("A3101")
for ALLELE in "${arr[@]}"
do
    for LEARNING_RATE in 0.01 0.001 0.0001 0.00001
    do
        for BATCH_SIZE in 40 60 70 80 100
        do
            for NUM_LAYERS in 1 2 3
            do
               python mhcPreds_tflearn_cmd_line.py -cmd  "train_test_eval" -b $BATCH_SIZE -e 30 -nl $NUM_LAYERS -bn 1 -d 32 -c 'one_hot' -a $ALLELE -m 'deep_rnn' --tensorboard-verbose 0 -r $LEARNING_RATE --run-id $ALLELE'_deep_rnn_layers_'$NUM_LAYERS'_batch_size_'$BATCH_SIZE'_LR_'$LEARNING_RATE
            done
        done
    done
done