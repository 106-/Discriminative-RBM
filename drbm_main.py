#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from DRBM import DRBM
import time

vector_size = 784
hidden_unit_num = 500
class_num = 10

class MNIST:
    def __init__(self, filename):
        self.data = []
        self.answer = []
        with open(filename, "r") as f:
            for line in f:
                sp_line = line.split(",")
                self.answer.append(DRBM.one_of_k(class_num, int(sp_line[0])))
                self.data.append(list(map(lambda x: int(x), sp_line[1:])))

def main():
    drbm = DRBM(vector_size, hidden_unit_num, class_num)
    train = MNIST("mnist_train.csv")
    test = MNIST("mnist_test.csv")

    print("train started.")
    start_time = time.time()
    drbm.train(train.data, train.answer, test.data, test.answer, 5000, test_interval=1)
    end_time = time.time()
    print("train end. time: {}sec".format(end_time-start_time))


if __name__=='__main__':
    main()