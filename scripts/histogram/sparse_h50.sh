#!/bin/sh

if [ $# -ne 2 ]; then
  echo "requires 2 arguments." 1>&2
  exit 1
fi

DIR="./results/histogram/"`date +%Y-%m-%d_%H-%M-%S`"_sparse_h50"
mkdir -p $DIR

data_num=500
epoch=3000
batch_size=50
learning_time=$(($data_num*$epoch/$batch_size))
# KLDの計算をしない
test_interval=$(($learning_time))

for file in `find ./params/v20h50c10_initial/ -type f | sort | sed -n $1,$2p`
do
    ./drbm_main.py $learning_time 0 ./settings/gengen/h50.json -s -l $data_num -i $test_interval -m $batch_size -d $DIR -g -a -p "s0_h50" -n $file
done
