#!/bin/sh

if [ $# -ne 2 ]; then
  echo "requires 2 arguments." 1>&2
  exit 1
fi

DIR="./results/mnist/d140_unlimit/"`date +%Y-%m-%d_%H-%M-%S`"_sparse_h100_MNIST_d140_unlimit"
mkdir -p $DIR

data_num=1000
epoch=100
batch_size=100
learning_time=$(($data_num*$epoch/$batch_size))
# 1epochごとにKLDやらの計算
test_interval=$(($data_num/$batch_size))

for file in `find ./params/v784h100c10_initial/ -type f | sort | sed -n $1,$2p`
do
    ./drbm_main.py $learning_time 0 ./settings/mnist_noise/d140_unlimit/h100.json -s -l $data_num -i $test_interval -m $batch_size -c -d $DIR -a -p "MNIST_sparse_h100" -n $file
done
