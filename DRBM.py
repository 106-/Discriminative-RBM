#!/usr/bin/env python
# -*- coding:utf-8 -*-

import math
import random
import numpy as np
import json
import logging
import datetime
from multiprocessing import Pool
import multiprocessing as mp
import numba

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

        self._old_diff_b = np.zeros(num_class)
        self._old_diff_v = np.zeros((num_visible, num_hidden))
        self._old_diff_c = np.zeros(num_hidden)
        self._old_diff_w = np.zeros((num_hidden, num_class))

        self.resume = None

    @staticmethod
    def _sigmoid(x):
        if -x > 708:
            return 0.0
        elif x > 708:
            return 1.0
        return 1/(1+np.exp(-x))

    @staticmethod
    def _log1p_exp(x):
        if x > 708:
            return x
        if 708 >= x > 0:
            return x + np.log1p( np.exp(-x) )
        elif 0 >= x > -708:
            return np.log1p( np.exp(x) )
        elif -708 >= x:
            return 0.0
        else:
            logging.debug(x)
            raise ValueError
    
    # one-of-k表現のベクトルを生成
    def one_of_k(self, number):
        # "One of K" -> "ok"
        ok = np.zeros(self.num_class)
        ok.put(number, 1)
        return ok

    # 31PにあるA_jを計算
    def _calc_A(self, j, input_vector, one_of_k):
        return self.bias_c[j] + np.dot(self.weight_w[j], one_of_k) + np.dot(self.weight_v[:,j], input_vector)

    @numba.jit
    def _get_A_matrix(self, input_vector):
        A_matrix = np.zeros((self.num_class, self.num_hidden))
        for k in range(self.num_class):
            for j in range(self.num_hidden):
                A_matrix[k][j] = self._calc_A(j, input_vector, self.one_of_k(k))
        return A_matrix
    
    @numba.jit
    def _get_probs_matrix(self, input_vectors):
        probs_matrix = np.zeros((len(input_vectors), self.num_class))
        for p in range(len(input_vectors)):
            probs_matrix[p] = self.probability(input_vectors[p])
        return probs_matrix
    
    def probability(self, input_vector, k=None):
        vlog1p_exp = np.vectorize(self._log1p_exp)
        A_matrix = self._get_A_matrix(input_vector)
        energies = self.bias_b + np.sum( vlog1p_exp(A_matrix) , axis=1)

        energy_max = np.max(energies)
        energies = np.exp(energies - energy_max)
        # nc -> Normalize Constant
        nc = np.sum(energies)
        probs = energies / nc

        if k == None:
            return probs
        else:
            return probs[k]

    @numba.jit
    def _get_A_matrix_diff(self, training):
        A_matrix = np.zeros((self.num_hidden, len(training.data), self.num_class))
        for j in range(self.num_hidden):
            for x in range(len(training.data)):
                for k in range(self.num_class):
                    A_matrix[j][x][k] = self._calc_A(j, training.data[x], self.one_of_k(k))
        return A_matrix

    def _differential_b(self, q, training):
        probs_matrix = self._get_probs_matrix(training.data)
        diff_b = np.sum(training.answer - probs_matrix, axis=0) / len(training.data)
        q.put(diff_b)
    
    def _differential_w(self, q, training):
        vsigmoid = np.vectorize(self._sigmoid)

        A_matrix = self._get_A_matrix_diff(training)
        probs_matrix = self._get_probs_matrix(training.data)

        A_mul_difftp = vsigmoid(A_matrix) * (training.answer - probs_matrix)
        diff_w = np.sum(A_mul_difftp, axis=1) / len(training.data)
        q.put(diff_w)
    
    def _differential_c(self, q, training):
        vsigmoid = np.vectorize(self._sigmoid)

        A_matrix = self._get_A_matrix_diff(training)
        probs_matrix = self._get_probs_matrix(training.data)
        sum_under_k = np.sum( vsigmoid(A_matrix) * probs_matrix, axis=2)

        A_matrix_tx = np.array([[self._calc_A(m, training.data[u], training.answer[u]) for u in range(len(training.data))] for m in range(self.num_hidden)])
        diff_c = np.sum( vsigmoid(A_matrix_tx) - sum_under_k, axis=1 ) / len(training.data)
        q.put(diff_c)
    
    def _differential_v(self, q, training):
        vsigmoid = np.vectorize(self._sigmoid)

        A_matrix = self._get_A_matrix_diff(training)
        probs_matrix = self._get_probs_matrix(training.data)
        sum_under_k = np.sum( vsigmoid(A_matrix) * probs_matrix, axis=2)

        A_matrix_tx = np.array([[self._calc_A(m, training.data[u], training.answer[u]) for u in range(len(training.data))] for m in range(self.num_hidden)])
        diff_v = np.dot( training.data.T, (vsigmoid(A_matrix_tx) - sum_under_k).T) / len(training.data)
        q.put(diff_v)

    def train(self, training, test, learning_time, batch_size, test_num_process, learning_rate=[0.01, 0.01, 0.1, 0.1], alpha=[0.9, 0.9, 0.9, 0.9], test_interval=100):
        if not (self.num_visible == len(training.data[0])):
            print(len(training.data[0]))
            raise TypeError
        
        resume_time = 0

        if not self.resume == None:
            resume_time = self.resume[0]
            learning_time = self.resume[1]

        for lt in range(resume_time, learning_time):
            try:
                batch = training.minibatch(batch_size)

                q = [mp.Queue() for i in range(4)]
                processes = [
                    mp.Process(target=self._differential_b, args=(q[0],batch)),
                    mp.Process(target=self._differential_w, args=(q[1],batch)),
                    mp.Process(target=self._differential_c, args=(q[2],batch)),
                    mp.Process(target=self._differential_v, args=(q[3],batch)),
                ]
                for p in processes:
                    p.start()
                diff_b = q[0].get()
                diff_w = q[1].get()
                diff_c = q[2].get()
                diff_v = q[3].get()
                for p in processes:
                    p.join()

                self.bias_b += diff_b * learning_rate[0] + self._old_diff_b * alpha[0]
                self.weight_w += diff_w * learning_rate[1] + self._old_diff_w * alpha[1]
                self.bias_c += diff_c * learning_rate[2] + self._old_diff_c * alpha[2]
                self.weight_v += diff_v * learning_rate[3] + self._old_diff_v * alpha[3]

                self._old_diff_b = diff_b
                self._old_diff_w = diff_w
                self._old_diff_c = diff_c
                self._old_diff_v = diff_v

                if lt % test_interval == 0:
                    self.test_error(test, test_num_process)
                    self.save("%d_of_%d.json"%(lt,learning_time), [lt, learning_time])
                    logging.info("parameters are dumpd.")

                logging.info("️training is processing. complete : {} / {}".format(lt+1, learning_time))

            except ValueError as e:
                logging.error(e)
                self.save("error.json")
                logging.error("error parameter saved.")
                exit()

            except KeyboardInterrupt as e:
                logging.info("train interrupted.")

                now = datetime.datetime.now()
                filename = now.strftime("%Y-%m-%d_%H-%M-%S.json")
                self.save(filename, [lt, learning_time])

                logging.info("parameters are dumped to %s"%filename)
                input_data = input('continue? (type "Y" to continue) : ')
                if input_data.upper() == "Y":
                    continue
                else:
                    exit()
    
    def classify(self, input_data):
        probs = self.probability(input_data)
        return np.argmax(probs)
    
    def test_error(self, test, num_process):
        correct = 0
        logging.info("correct rate caluculating.")
        splitted_datas = np.array_split(test.data, num_process)
        splitted_answers = np.array_split(test.answer, num_process)
        with Pool(processes=num_process) as pool:
            correct = np.sum(pool.map(self._count_correct, list(zip(splitted_datas, splitted_answers))))
        
        correct_rate = correct / float(len(test.data))
        logging.info("️correct rate: {} ({} / {})".format( correct_rate, correct, len(test.data) ))

    @numba.jit
    def _count_correct(self, args):
        classified_data = np.zeros((len(args[0]), self.num_class))
        for u in range(len(args[0])):
            classified_data[u] = self.one_of_k(self.classify(args[0][u]))
        correct = int(np.sum( args[1] * classified_data ))
        return correct

    def save(self, filename, training_progress=None):
        params = {
            "training_progress":training_progress,
            "num_class":self.num_class,
            "num_hidden":self.num_hidden,
            "num_visible":self.num_visible,
            "bias_b":self.bias_b.tolist(),
            "bias_c":self.bias_c.tolist(),
            "weight_w":self.weight_w.tolist(),
            "weight_v":self.weight_v.tolist(),
        }
        if not training_progress == None:
            params["training_progress"] = training_progress
        json.dump(params, open(filename, "w+"))
    
    @staticmethod
    def load_from_json(filename):
        params = json.load(open(filename, "r"))
        drbm = DRBM(params["num_visible"], params["num_hidden"], params["num_class"])
        drbm.bias_b = np.array(params["bias_b"])
        drbm.bias_c = np.array(params["bias_c"])
        drbm.weight_w = np.array(params["weight_w"])
        drbm.weight_v = np.array(params["weight_v"])
        drbm.resume = params["training_progress"]
        return drbm

def main():
    drbm = DRBM(3, 5, 7)
    a = drbm.probability(np.array([123,56,23]))
    print(a)
    print(sum(a))

if __name__=='__main__':
    main()