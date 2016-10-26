#!/bin/bash

set -x # echo all commands executed

mkdir -p logs

for layers in $(seq 1 10)
do
    date
    TS=$(date "+%Y%m%d-%H%M%S")
    python hw2b.py --n_layers $layers --n_hidden 500 -e1000 &> logs/run-${layers}x500-${TS}.log
done

