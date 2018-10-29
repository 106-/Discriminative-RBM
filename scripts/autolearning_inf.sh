#!/bin/sh

for i in `seq 1 20`
do
    ./drbm_main.py 10000 0 -l 1000 2>&1 | tee ./results/inf-${i}.log
done