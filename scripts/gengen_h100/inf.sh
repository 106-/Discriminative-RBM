#!/bin/sh

if [ $# -ne 2 ]; then
  echo "requires 2 arguments." 1>&2
  exit 1
fi

DIR="./results/sgd_vs_adamax/generated_generative_model_h100/"`date +%Y-%m-%d_%H-%M-%S`"_inf_v20h100c10"
mkdir -p $DIR

data_num=500
epoch=3000
batch_size=50
learning_time=$(($data_num*$epoch/$batch_size))
# 1epochごとにKLDやらの計算
test_interval=$(($data_num/$batch_size))

for file in `find ./params/v20h100c10_initial/ -type f | sort | sed -n $1,$2p`
do
    ./drbm_main.py $learning_time 0 ./settings/gengen/h100.json -l $data_num -i $test_interval -m $batch_size -d $DIR -g -p "inf" -n $file
done
