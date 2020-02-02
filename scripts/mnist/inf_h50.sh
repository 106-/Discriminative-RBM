#!/bin/sh

if [ $# -ne 2 ]; then
  echo "requires 2 arguments." 1>&2
  exit 1
fi

DIR="./results/mnist/"`date +%Y-%m-%d_%H-%M-%S`"_inf_MNIST"
mkdir -p $DIR

data_num=1000
epoch=100
batch_size=100
learning_time=$(($data_num*$epoch/$batch_size))
# 1epochごとにKLDやらの計算
test_interval=$(($data_num/$batch_size))

for file in `find ./params/v784h50c10_initial/ -type f | sort | sed -n $1,$2p`
do
    ./drbm_main.py $learning_time 0 ./settings/MNIST_noise_h50.json -l $data_num -i $test_interval -m $batch_size -c -d $DIR -p "MNIST_inf_h50" -n $file
done
