#!/bin/bash

set -x # echo all commands executed

mkdir -p logs


# for i in $(seq 1 7); do j=$(echo 420/$i | bc); echo ${i}x${j}; done;

for total_hidden in $(seq 420 420 840)
do
    for layers in $(seq 1 10)
    do
        date
        TS=$(date "+%Y%m%d-%H%M%S")
        hidden=$(echo $total_hidden/$layers | bc)
        LOG_FILE=logs/run-2b4-${layers}x${hidden}-${TS}.log
        date
        echo 'start '$(date) >> $LOG_FILE 
        python hw2b.py --n_layers $layers --n_hidden $hidden -e1000 -v &> $LOG_FILE
        echo 'end '$(date) >> $LOG_FILE
    done
done
