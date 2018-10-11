#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from DRBM import DRBM
import time
import numpy as np
import logging
import argparse

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
np.seterr(over="raise", invalid="raise")

parser = argparse.ArgumentParser(description="DRBM learning script")
parser.add_argument("learning_num", action="store", type=int, help="number of updating parameters")
parser.add_argument("division_num", action="store", type=int, help="number of dividing middle layer")
parser.add_argument("-m", "--minibatch_size", action="store", default=100, type=int, help="minibatch size")
args = parser.parse_args()

class LearningData:
    def __init__(self, data, answer):
        self.data = data
        self.answer = answer

    def minibatch(self, batchsize, random=True):
        batch_data = None
        batch_answer = None
        if random:
            length = len(self.data)
            idx = np.random.choice(np.arange(0, length), size=batchsize, replace=False)
            batch_data = np.take(self.data, idx, axis=0)
            batch_answer = np.take(self.answer, idx, axis=0)
        else:
            batch_data = self.data[:batchsize]
            batch_answer = self.answer[:batchsize]
        return LearningData(batch_data, batch_answer)

class MNIST(LearningData):
    def __init__(self, filename, num_class):
        array = np.load(filename)
        self.answer, self.data = np.split(array, [1], axis=1)
        self.data = self.data.astype("float64") / 255
        self.answer = np.eye(num_class)[self.answer.flatten().tolist()]
    
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
    hidden_unit_num = 200
    class_num = 10
    hidden_layer_value_num = args.division_num

    logging.info("input_vector(n):%d, hidden_unit(m):%d, class_num(K):%d, div_num:%d"%(vector_size, hidden_unit_num, class_num, hidden_layer_value_num))

    initial_parameters = {
        "weight_w": np.load("weight_w.npy"),
        "weight_v": np.load("weight_v.npy"),
        "bias_c": np.load("bias_c.npy"),
        "bias_b": np.load("bias_b.npy"),
    }

    drbm = DRBM(vector_size, hidden_unit_num, class_num, hidden_layer_value_num, initial_parameter=initial_parameters)

    logging.info("️start loading data.")
    train = MNIST("mnist_train.npy", class_num)
    test = MNIST("mnist_test.npy", class_num)
    logging.info("☑ loading data complete.")

    logging.info("train started.")
    start_time = time.time()

    drbm.train(train, test, args.learning_num, args.minibatch_size, learning_rate=[0.1, 0.1, 1.0, 1.0], calc_train_correct_rate=True)
    
    end_time = time.time()
    logging.info("☑ train complete. time: {} sec".format(end_time-start_time))

    drbm.save("parameters.json")
    logging.info("☑ parameters dumped.")


if __name__=='__main__':
    main()