#!/bin/sh

cd $1

for i in `seq 1 20`
do
    cat inf-${i}.log | grep "train correct rate" > cr-train-inf-${i}.txt
    cat inf-${i}.log | grep "test correct rate" > cr-test-inf-${i}.txt
    cat 2-${i}.log | grep "train correct rate" > cr-train-2-${i}.txt
    cat 2-${i}.log | grep "test correct rate" > cr-test-2-${i}.txt
done
