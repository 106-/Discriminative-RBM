#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import numpy as np

parser = argparse.ArgumentParser("")
parser.add_argument("-m", "--mean", action="store", type=float, help="mean of noise")
parser.add_argument("-d", "--deviation", action="store", type=float, help="deviation of noise")
args = parser.parse_args()

def main():
    #train = np.load("mnist_train.npy")
    test = np.load("mnist_test.npy")
    #train_answer, train_data = np.split(train.astype("float64"), [1], axis=1)
    #train_data /= 255
    test_answer, test_data = np.split(test.astype("float64"), [1], axis=1)
    test_data /= 255

    mean = args.mean
    dev = args.deviation

    #train_data += np.random.normal(mean, dev, train_data.shape)
    test_data += np.random.normal(mean, dev, test_data.shape)

    #train_data = np.clip(train_data, 0, 1)
    test_data = np.clip(test_data, 0, 1)

    #noised_train = np.hstack((train_answer, train_data))
    noised_test= np.hstack((test_answer, test_data))

    #np.save("mnist_noise_train", noised_train)
    np.save("mnist_noise_test", noised_test)

if __name__=='__main__':
    main()