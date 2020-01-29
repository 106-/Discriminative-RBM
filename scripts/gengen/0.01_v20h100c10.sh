#!/bin/sh

DIR="./results/"`date +%Y-%m-%d_%H-%M-%S`"_gengen_0.01_v20h100c10_sampling"
mkdir -p $DIR

data_num=500
epoch=3000
batch_size=50
learning_time=$(($data_num*$epoch/$batch_size))
# 1epochごとにKLDやらの計算
test_interval=$(($data_num/$batch_size))

for file in `find ./params/v20h100c10_initial/ -type f | sort`
do
    ./drbm_main.py $learning_time 0 ./settings/gengen_v20h100c10.json -s -l $data_num -i $test_interval -m $batch_size -d $DIR -g -r 0.01 -p "0.01" -n $file
done
