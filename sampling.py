#!/usr/bin/env python
# -*- coding:utf-8 -*-

from DRBM import DRBM
import argparse
import numpy as np

np.seterr(over="raise", invalid="raise")

parser = argparse.ArgumentParser("DRBM sampling script", add_help=False)
parser.add_argument("-o", "--output", action="store", default="output", type=str, help="output filename")
parser.add_argument("-v", "--num_visible", action="store", default=300, type=int, help="number of visible layer") 
parser.add_argument("-c", "--num_class", action="store", default=10, type=int, help="number of class layer") 
parser.add_argument("-h", "--num_hidden", action="store", default=200, type=int, help="number of hidden layer") 
parser.add_argument("-p", "--drbm_parameters", action="store", default=None, type=str, help="json filename of DRBM parameters") 
parser.add_argument("division_num", action="store", type=int, help="number of dividing middle layer")
parser.add_argument("sampling_num", action="store", type=int, help="number of sampling")
args = parser.parse_args()

def main():
    v = args.num_visible
    h = args.num_hidden
    c = args.num_class
    d = args.division_num
    p = args.drbm_parameters

    drbm = None
    if p is None:
        drbm = DRBM(v, h, c, d, random_bias=True)
        drbm.save("sampling_parameters.json")
    else:
        drbm = DRBM.load_from_json(p)

    random_data = np.random.randn(args.sampling_num, drbm.num_visible)
    answers = None 
    split_num = np.ceil( args.sampling_num /10000 )
    splitted_datas = np.array_split(random_data, split_num, axis=0)
    for d in splitted_datas:
        a_matrix = drbm.matrix_ok_A(d)
        probs = drbm.probability(a_matrix)
        np.cumsum(probs, axis=1, out=probs)
        uniform_dist = np.random.rand(len(d),1)
        np.subtract(probs, uniform_dist, out=probs)
        np.less(0, probs, out=probs)
        random_class = np.argmax(probs, axis=1)

        if answers is not None:
            answers = np.hstack((answers, random_class))
        else:
            answers = random_class
    
    output = np.concatenate((answers[:, np.newaxis], random_data), axis=1)

    np.save(args.output, output)

if __name__=='__main__':
    main()