#!/bin/sh

if [ $# -ne 2 ]; then
  echo "requires 2 arguments." 1>&2
  exit 1
fi

DIR="./results/"`date +%Y-%m-%d_%H-%M-%S`"_d$1_$2_noise"
mkdir -p $DIR

for i in `seq $2`
do
    ./drbm_main.py 5000 $1 ./settings/MNIST_noise.json -l 1000 -t -o adam -d $DIR
done