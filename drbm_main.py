#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from DRBM import DRBM
import time
import numpy as np
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
np.seterr(over="raise", invalid="raise")

class LearningData:
    def __init__(self, data, answer):
        self.data = data
        self.answer = answer

    def minibatch(self, batchsize):
        length = len(self.data)
        idx = np.random.choice(np.arange(0, length), size=batchsize, replace=False)
        batch_data = np.take(self.data, idx, axis=0)
        batch_answer = np.take(self.answer, idx, axis=0)
        return LearningData(batch_data, batch_answer)

class MNIST(LearningData):
    def __init__(self, filename, num_class):
        self.data = []
        self.answer = []
        with open(filename, "r") as f:
            for line in f:
                sp_line = line.split(",")
                self.answer.append(one_of_k(num_class, int(sp_line[0])))
                self.data.append(list(map(lambda x: int(x), sp_line[1:])))
        self.answer = np.array(self.answer)
        self.data = np.array(self.data)
    
class dummy_data(LearningData):
    def __init__(self, data_num, num_input, num_class):
        self.answer = [one_of_k(num_class, i) for i in np.random.randint(0, num_class, data_num)]
        self.data = np.random.randint(0, 256, (data_num, num_input))

# one-of-k表現のベクトルを生成
def one_of_k(num_class, number):
    # "One of K" -> "ok"
    ok = np.zeros(num_class)
    ok.put(number, 1)
    return ok

def main():
    vector_size = 784
    hidden_unit_num = 500
    class_num = 10

    drbm = DRBM(vector_size, hidden_unit_num, class_num)

    # logging.info("creating dummy data.")
    # train = dummy_data(100, vector_size, class_num)
    # test = dummy_data(100, vector_size, class_num)
    # logging.info("☑ creating dummy data complete.")

    logging.info("️start loading data.")
    train = MNIST("mnist_train.csv", class_num)
    test = MNIST("mnist_test.csv", class_num)
    logging.info("☑ loading data complete.")

    logging.info("train started.")
    start_time = time.time()

    drbm.train(train, test, 5000, 100, 8)
    
    end_time = time.time()
    logging.info("☑ train complete. time: {} sec".format(end_time-start_time))

    drbm.save("parameters.json")
    logging.info("☑ parameters dumped.")


if __name__=='__main__':
    main()