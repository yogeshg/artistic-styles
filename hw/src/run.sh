#!/bin/bash

set -x # echo all commands executed

mkdir -p logs
TS=$(date "+%Y%m%d-%H%M%S")
for layers in $(seq 1 2 7)
do
    date
    hidden=500
    python hw2b.py --n_layers $layers --n_hidden $hidden -e1000 -v &> logs/run-${layers}x${hidden}-${TS}.log
done

