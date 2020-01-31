#!/bin/sh

if [ $# -ne 2 ]; then
  echo "requires 2 arguments." 1>&2
  exit 1
fi

DIR="./results/kld/"`date +%Y-%m-%d_%H-%M-%S`"_sparse_v20h200c10"
mkdir -p $DIR

data_num=500
epoch=3000
batch_size=50
learning_time=$(($data_num*$epoch/$batch_size))
# 1epochごとにKLDやらの計算
test_interval=$(($data_num/$batch_size))

for file in `find ./params/v20h200c10_initial/ -type f | sort | sed -n $1,$2p`
do
    ./drbm_main.py $learning_time 0 ./settings/gengen_v20h200c10.json -s -l $data_num -i $test_interval -m $batch_size -d $DIR -g -a -p "s0_h200" -n $file
done
