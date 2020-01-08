#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from DRBM import DRBM
from DRBM import parameters
import optimizer
import time
import numpy as np
import logging
import argparse
import json
import os
import datetime

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
np.seterr(over="raise", invalid="raise")
args = None

def arg_setting():
    global args
    parser = argparse.ArgumentParser(description="DRBM learning script")
    parser.add_argument("learning_num", action="store", type=int, help="number of updating parameters")
    parser.add_argument("division_num", action="store", type=int, help="number of dividing middle layer")
    parser.add_argument("train_setting_file", action="store", type=str, help="json file which settings parameters")
    parser.add_argument("-l", "--datasize_limit", action="store", default=0, type=int, help="limit data size")
    parser.add_argument("-m", "--minibatch_size", action="store", default=100, type=int, help="minibatch size")
    parser.add_argument("-o", "--optimizer", action="store", default="adamax", type=str, help="optimizer")
    parser.add_argument("-c", "--correct_rate", action="store_true", help="calculate correct rate for training|test data or not")
    parser.add_argument("-k", "--kl_divergence", action="store", type=str, default=None, help="calculate kl-divergence from specified DRBM (json file)")
    parser.add_argument("-i", "--test_interval", action="store", type=int, default=100, help="interval to calculate ( test error | training error | KL-Divergence )")
    parser.add_argument("-d", "--result_directory", action="store", type=str, default="./results/", help="directory to output learning result file.")
    parser.add_argument("-s", "--sparse", action="store_true", help="enable sparse normalization or not")
    args = parser.parse_args()

class LearningData:
    def __init__(self, data, answer):
        self.data = data
        self.answer = answer
        self.minibatch_buffer = np.array([])

    def restore_minibatch(self, batchsize, random=True):
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
    
    def minibatch(self, batchsize):
        if len(self.minibatch_buffer)==0:
            self.minibatch_buffer = np.random.choice(np.arange(0, len(self.data)), size=len(self.data), replace=False)
        idx, self.minibatch_buffer = np.split(self.minibatch_buffer, [batchsize])
        batch_data = self.data[idx]
        batch_answer = self.answer[idx]
        return LearningData(batch_data, batch_answer)
    
    def normalize(self, mu=None, sigma=None):
        if mu is None:
            mu = np.mean(self.data, axis=0)
        if sigma is None:
            sigma = np.std(self.data, axis=0)
            sigma[sigma == 0] = 1
        self.data = (self.data - mu) / sigma
        np.clip(self.data, -4, 4, out=self.data)
        return mu, sigma
    
    def normalize_255(self):
        self.data /= 255

class MNIST(LearningData):
    def __init__(self, filename, num_class):
        array = np.load(filename)
        self.answer_origin, data = np.split(array, [1], axis=1)
        data = data.astype("float64")
        answer = np.eye(num_class)[self.answer_origin.astype("int64").flatten().tolist()]
        super(MNIST, self).__init__(data, answer)
    
    def save(self, filename):
        stacked = np.hstack((self.answer_origin, self.data))
        np.save(filename, stacked)
    
class LearningDataSettings:
    def __init__(self, filename):
        with open(filename, "r") as f:
            json_setting = json.load(f)
            self.input_unit = json_setting["input-unit"]
            self.hidden_unit = json_setting["hidden-unit"]
            self.class_unit = json_setting["class-unit"]
            self.training_data = MNIST(json_setting["training-data"], self.class_unit)
            self.test_data = MNIST(json_setting["test-data"], self.class_unit)
            self.initial_model = json_setting["initial-model"]
            if json_setting["needs-normalize"]:
                # mu, sigma = self.training_data.normalize()
                # self.test_data.normalize(mu=mu, sigma=sigma)
                self.training_data.normalize_255()
                self.test_data.normalize_255()

def main():
    logging.info("️start loading setting data.")
    settings = LearningDataSettings(args.train_setting_file)
    logging.info("☑ loading setting data complete.")

    vector_size = settings.input_unit
    hidden_unit_num = settings.hidden_unit
    class_num = settings.class_unit
    hidden_layer_value_num = args.division_num

    logging.info("input_vector(n):%d, hidden_unit(m):%d, class_num(K):%d, div_num:%d"%(vector_size, hidden_unit_num, class_num, hidden_layer_value_num))
    
    drbm = DRBM.load_from_json(settings.initial_model, args.division_num, args.sparse)

    if args.datasize_limit != 0:
        settings.training_data = settings.training_data.restore_minibatch(args.datasize_limit, random=False)

    gen_drbm = None
    if args.kl_divergence:
        gen_drbm = DRBM.load_from_json(args.kl_divergence)

    opt = None
    if args.optimizer == "momentum":
        logging.info("optimize method: momentum")
        opt = optimizer.momentum(vector_size, hidden_unit_num, class_num)
    elif args.optimizer == "adam":
        logging.info("optimize method: adam")
        opt = optimizer.adam(vector_size, hidden_unit_num, class_num)
    else:
        logging.info("optimize method: adamax")
        opt = optimizer.adamax(vector_size, hidden_unit_num, class_num)

    logging.info("train started.")
    start_time = time.time()

    learning_result = drbm.train(settings.training_data, settings.test_data, args.learning_num, args.minibatch_size, opt, test_interval=args.test_interval,
        correct_rate=args.correct_rate,
        gen_drbm=gen_drbm,
    )
    
    end_time = time.time()
    logging.info("☑ train complete. time: {} sec".format(end_time-start_time))

    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    hidden_layer = "s" if args.sparse else "d"
    filename_template = "{}_{}{}_v{}h{}c{}_%s.json".format(now, hidden_layer, drbm.div_num, drbm.num_visible, drbm.num_hidden, drbm.num_class)
    learning_result.save(os.path.join(args.result_directory, filename_template%"log"))

    drbm.save( os.path.join(args.result_directory, filename_template%"params"))
    logging.info("☑ parameters dumped.")


if __name__=='__main__':
    arg_setting()
    main()