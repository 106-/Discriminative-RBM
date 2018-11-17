#!/bin/sh

if [ $# -ne 2 ]; then
  echo "requires 2 arguments." 1>&2
  exit 1
fi

DIR="./results/"`date +%Y-%m-%d_%H-%M-%S`"_d$1_$2_sampling"
mkdir -p $DIR

for i in `seq $2`
do
    ./drbm_main.py 1000 $1 ./settings/sampling_v20h50c10.json -l 1000 -i 10 -t -k ./datas/v20h50c10.json -d $DIR
done