#!/bin/sh

if [ $# -ne 1 ]; then
  echo "requires 1 arguments." 1>&2
  exit 1
fi

DIR="./results/"`date +%Y-%m-%d_%H-%M-%S`"_pure_MNIST"
mkdir -p $DIR

data_num=60000
epoch=100
batch_size=100
learning_time=$(($data_num*$epoch/$batch_size))
# 1epochごとにKLDやらの計算
test_interval=$(($data_num/$batch_size))

for i in `seq $1`
do
    ./drbm_main.py $learning_time 0 ./settings/mnist/mnist.json -l $data_num -i $test_interval -m $batch_size -c -d $DIR
done