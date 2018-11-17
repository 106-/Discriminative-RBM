#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import numpy as np
from drbm_main import MNIST
# import matplotlib.pyplot as plt

parser = argparse.ArgumentParser("")
parser.add_argument("deviation", action="store", type=float, help="deviation of noise")
args = parser.parse_args()

def main():
    test = MNIST("./datas/mnist_test.npy", 10)
    dev = args.deviation
    np.add(test.data, np.random.normal(0, dev, test.data.shape), out=test.data)
    np.clip(test.data, 0, 255, out=test.data)
    # plt.hist(test.data[0], bins=255)
    # plt.show()
    test.save("./datas/mnist_noise_test_d{}".format(dev))

if __name__=='__main__':
    main()