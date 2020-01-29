#!/bin/sh

if [ $# -ne 1 ]; then
  echo "requires 1 arguments." 1>&2
  exit 1
fi

DIR="./results/"`date +%Y-%m-%d_%H-%M-%S`"_N$1_inf_v20h200c10_sampling"
mkdir -p $DIR

data_num=500
epoch=1000
batch_size=100
learning_time=$(($data_num*$epoch/$batch_size))
# 1epochごとにKLDやらの計算
test_interval=$(($data_num/$batch_size))

for i in `seq $1`
do
    ./drbm_main.py $learning_time 0 ./settings/v20h200c10.json -l $data_num -i $test_interval -m $batch_size -k ./params/gen_inf_v20h50c10.json -d $DIR
done
