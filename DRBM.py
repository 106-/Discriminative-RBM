#!/usr/bin/env python
# -*- coding:utf-8 -*-

import math
import random
import numpy as np
import json
import logging
import datetime
import traceback
from multiprocessing import Pool
import multiprocessing as mp

class DRBM:

    def __init__(self, num_visible, num_hidden, num_class, div_num):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.num_class = num_class
        self.div_num = div_num
        self.div_factors = np.linspace(-1, 1, self.div_num)
        # Xavierの初期値
        sq_node = 1 / math.sqrt(max(num_visible, num_hidden, num_class))
        self.weight_v = sq_node * np.random.randn(self.num_visible, self.num_hidden)
        self.weight_w = sq_node * np.random.randn(self.num_class, self.num_hidden)  
        self.bias_c = sq_node * np.random.randn(self.num_hidden)
        self.bias_b = sq_node * np.random.randn(self.num_class)

        self._old_diff_b = np.zeros(num_class)
        self._old_diff_v = np.zeros((num_visible, num_hidden))
        self._old_diff_c = np.zeros(num_hidden)
        self._old_diff_w = np.zeros((num_class, num_hidden))

        self.vlog1p_exp = np.vectorize(self._log1p_exp)
        self.vsigmoid = np.vectorize(self._sigmoid)
        self.vmarginal_prob = np.vectorize(self._marginal_prob)
        self.vmarginal = np.vectorize(self._marginal)

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
    
    def _marginal(self,x):
        args = x * self.div_factors
        denomi = np.sum(np.cosh(args))
        nume = np.dot(self.div_factors, np.sinh(args))
        return nume / denomi
    
    def _marginal_prob(self,x):
        return np.log(np.sum(np.exp(self.div_factors * x)))

    # 全クラス分/全データ分のA_jを計算(N,K,m)のサイズ
    def _matrix_ok_A(self, input_vector):
        return self.bias_c + self.weight_w + np.dot(input_vector, self.weight_v)[:, np.newaxis, :]

    # データに対してのA_j (N,m)
    def _matrix_A(self, answer, data):
        return self.bias_c + np.dot(answer, self.weight_w) + np.dot(data, self.weight_v)

    # N個のデータに対して(N,K)の確率の行列を返す
    def probability(self, A_matrix, normalize=True):
        # energies = self.bias_b + np.sum( self.vlog1p_exp(A_matrix) , axis=2)
        energies = self.bias_b + np.sum( self.vmarginal_prob(A_matrix) , axis=2)

        if normalize:
            max_energy = np.max(energies)
            energies = np.exp(energies-max_energy)
            return energies / np.sum(energies, axis=1)[:, np.newaxis]
        else:
            return energies

    # b,wの勾配を計算diff_bは(K),diff_wは(K,m)
    def _differential_bw(self, q, training):
        diff_tp = training.answer - self._probs_matrix
        diff_b = np.sum(diff_tp, axis=0) / len(training.data)
        diff_w = np.sum(np.multiply(self._A_matrix_ok, diff_tp[:, :, np.newaxis]), axis=0) / len(training.data)
        q.put(diff_b)
        q.put(diff_w)
    
    # c,vの勾配 diff_cは(m), diff_vは(n,m)
    def _differential_cv(self, q, training):
        sum_under_k = np.sum( self._sig_A_ok * self._probs_matrix[:, :, np.newaxis], axis=1)
        diff_sig_a = self._sig_A - sum_under_k
        diff_c = np.sum( diff_sig_a, axis=0 ) / len(training.data)
        diff_v = np.sum( np.multiply( training.data[:, :, np.newaxis], diff_sig_a[:, np.newaxis, :]), axis=0) / len(training.data)
        q.put(diff_c)
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
                self._A_matrix_ok = self._matrix_ok_A(batch.data)
                self._A_matrix = self._matrix_A(batch.answer, batch.data)
                self._probs_matrix = self.probability(self._A_matrix_ok)
                self._sig_A_ok = self.vmarginal(self._A_matrix_ok)
                self._sig_A = self.vmarginal(self._A_matrix)

                q = [mp.Queue() for i in range(2)]
                # self._differential_bw(q[0], batch)
                # self._differential_cv(q[1], batch)
                processes = [
                    mp.Process(target=self._differential_bw, args=(q[0],batch)),
                    mp.Process(target=self._differential_cv, args=(q[1],batch)),
                ]
                for p in processes:
                    p.start()
                diff_b = q[0].get()
                diff_w = q[0].get()
                diff_c = q[1].get()
                diff_v = q[1].get()
                for p in processes:
                    p.join()

                self.bias_b   += learning_rate[0] * (diff_b + self._old_diff_b * alpha[0])
                self.weight_w += learning_rate[1] * (diff_w + self._old_diff_w * alpha[1])
                self.bias_c   += learning_rate[2] * (diff_c + self._old_diff_c * alpha[2])
                self.weight_v += learning_rate[3] * (diff_v + self._old_diff_v * alpha[3])

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
                logging.error(traceback.format_exc())
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
        probs = self.probability(self._matrix_ok_A(input_data), normalize=False)
        return np.argmax(probs, axis=1)
    
    def test_error(self, test, num_process):
        correct = 0
        logging.info("correct rate caluculating.")
        splitted_datas = np.array_split(test.data, num_process, axis=0)
        splitted_answers = np.array_split(test.answer, num_process, axis=0)
        with Pool(processes=num_process) as pool:
            correct = np.sum(pool.map(self._count_correct, list(zip(splitted_datas, splitted_answers))))
        
        correct_rate = correct / float(len(test.data))
        logging.info("️correct rate: {} ({} / {})".format( correct_rate, correct, len(test.data) ))

    def _count_correct(self, args):
        answers = np.dot( args[1], np.arange(self.num_class) )
        classified_data = np.where( answers == self.classify(args[0]), 1, 0)
        correct = np.sum( classified_data )
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
    print(a)
    print(sum(a))

if __name__=='__main__':
    main()