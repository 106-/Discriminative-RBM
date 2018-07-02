#!/usr/bin/env python
# -*- coding:utf-8 -*-

import math
import random
import numpy as np
import json
import logging

class DRBM:

    def __init__(self, num_visible, num_hidden, num_class):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.num_class = num_class
        # Xavierの初期値
        sq_node = 1 / math.sqrt(max(num_visible, num_hidden, num_class))
        self.weight_v = sq_node * np.random.randn(self.num_visible, self.num_hidden)
        self.weight_w = sq_node * np.random.randn(self.num_hidden, self.num_class) 
        self.bias_c = sq_node * np.random.randn(self.num_hidden)
        self.bias_b = sq_node * np.random.randn(self.num_class)

    @staticmethod
    def _sigmoid(x):
        # オーバーフロー対策
        if -x > 709:
            return 1.2167807506234229e-308
        return 1/(1+np.exp(-x))

    @staticmethod
    def _exp(x):
        if x>0:
            return x + np.log1p( np.exp(-x) )
        else:
            return np.log1p( np.exp(x) )

    # one-of-k表現のベクトルを生成
    def one_of_k(self, number):
        # "One of K" -> "ok"
        ok = np.zeros(self.num_class)
        ok.put(number, 1)
        return ok

    # 31PにあるA_jを計算
    def _calc_A(self, j, input_vector, one_of_k):
        return self.bias_c[j] * np.dot(self.weight_w[j], one_of_k) * np.dot(self.weight_v[:,j], input_vector)

    def probability(self, input_vector, k=None):
        vexp = np.vectorize(self._exp)
        A_matrix = np.array([[self._calc_A(m, input_vector, self.one_of_k(k)) for m in range(self.num_hidden)] for k in range(self.num_class)])
        log_a = np.log( np.ones((self.num_class, self.num_hidden)) + vexp(-A_matrix) )
        sum_under_j = np.sum( A_matrix + log_a , axis=1)

        energies = vexp( self.bias_b + sum_under_j )
        # nc -> Normalize Constant
        nc = np.sum(energies)
        probs = energies / nc

        if k == None:
            return probs
        else:
            return probs[k]

    def _differential_b(self, training_datas, training_answers):
        probs_matrix = np.array([self.probability(x) for x in training_datas])
        diff_b = np.sum(training_answers - probs_matrix, axis=0) / len(training_datas)
        return diff_b
    
    def _differential_w(self, training_datas, training_answers):
        vsigmoid = np.vectorize(self._sigmoid)

        A_matrix = np.array([[[self._calc_A(m, x, self.one_of_k(k)) for k in range(self.num_class)] for x in training_datas] for m in range(self.num_hidden)])
        probs_matrix = np.array([self.probability(x) for x in training_datas])

        A_mul_difftp = vsigmoid(A_matrix) * (training_answers - probs_matrix)
        diff_w = np.sum(A_mul_difftp, axis=1) / len(training_datas)
        return diff_w
    
    def _differential_c(self, training_datas, training_answers):
        vsigmoid = np.vectorize(self._sigmoid)

        A_matrix = np.array([[[self._calc_A(m, x, self.one_of_k(k)) for k in range(self.num_class)] for x in training_datas] for m in range(self.num_hidden)])
        probs_matrix = np.array([self.probability(x) for x in training_datas])
        sum_under_k = np.sum(A_matrix * probs_matrix, axis=2)

        A_matrix_tx = np.array([[self._calc_A(m, training_datas[u], training_answers[u]) for u in range(len(training_datas))] for m in range(self.num_hidden)])
        diff_c = np.sum( vsigmoid(A_matrix_tx) - sum_under_k, axis=1 ) / len(training_datas)
        return diff_c
    
    def _differential_v(self, training_datas, training_answers):
        vsigmoid = np.vectorize(self._sigmoid)

        A_matrix = np.array([[[self._calc_A(m, x, self.one_of_k(k)) for k in range(self.num_class)] for x in training_datas] for m in range(self.num_hidden)])
        probs_matrix = np.array([self.probability(x) for x in training_datas])
        sum_under_k = np.sum(A_matrix * probs_matrix, axis=2)

        A_matrix_tx = np.array([[self._calc_A(m, training_datas[u], training_answers[u]) for u in range(len(training_datas))] for m in range(self.num_hidden)])
        diff_v = np.dot( training_datas.T, (vsigmoid(A_matrix_tx) - sum_under_k).T)
        return diff_v

    def train(self, training, test, learning_time, batch_size, learning_rate=[0.01, 0.01, 0.01, 0.01], test_interval=100):
        if not (self.num_visible == len(training.data[0])):
            print(len(training.data[0]))
            raise TypeError
        
        for lt in range(learning_time):
            batch = training.minibatch(batch_size)

            diff_b = self._differential_b(batch.data, batch.answer)
            diff_w = self._differential_w(batch.data, batch.answer)
            diff_c = self._differential_c(batch.data, batch.answer)
            diff_v = self._differential_v(batch.data, batch.answer)

            self.bias_b += diff_b * learning_rate[0]
            self.weight_w += diff_w * learning_rate[1]
            self.bias_c += diff_c * learning_rate[2]
            self.weight_v += diff_v * learning_rate[3]
            
            logging.info("️training is processing. {} / {}".format(lt+1, learning_time))
            if lt % test_interval == 0:
                self.test_error(test)
    
    def classify(self, input_data):
        probs = self.probability(input_data)
        return np.argmax(probs)
    
    def test_error(self, test):
        classified_data = [self.one_of_k(self.classify(d)) for d in test.data]
        correct = np.sum( np.dot(test.answer, classified_data.T) )
        logging.info("️correct rate: {}".format(correct / float(len(test.data))))

    def save(self, filename):
        params = {
            "num_class":self.num_class,
            "num_hidden":self.num_hidden,
            "num_visible":self.num_visible,
            "bias_b":self.bias_b.tolist(),
            "bias_c":self.bias_c.tolist(),
            "weight_w":self.weight_w.tolist(),
            "weight_v":self.weight_v.tolist(),
        }
        json.dump(params, open(filename, "w+"))
    
    @staticmethod
    def load_from_json(filename):
        params = json.load(open(filename, "r"))
        drbm = DRBM(params["num_visible"], params["num_hidden"], params["num_class"])
        drbm.bias_b = np.array(params["bias_b"])
        drbm.bias_c = np.array(params["bias_c"])
        drbm.weight_w = np.array(params["weight_w"])
        drbm.weight_v = np.array(params["weight_v"])
        return drbm

def main():
    drbm = DRBM(3, 5, 7)
    a = drbm.probability(np.array([123,56,23]))
    print(a)
    print(sum(a))

if __name__=='__main__':
    main()