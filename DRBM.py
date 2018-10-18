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


class parameters:
    def __init__(self, num_visible, num_hidden, num_class, randominit=True, initial_parameter=None):
        if randominit:
            # Xavierの初期値
            sq_node = 1 / math.sqrt(max(num_visible, num_hidden, num_class))
            self.weight_v = sq_node * np.random.randn(num_visible, num_hidden)
            self.weight_w = sq_node * np.random.randn(num_class, num_hidden)  
            self.bias_c = sq_node * np.random.randn(num_hidden)
            self.bias_b = sq_node * np.random.randn(num_class)
        elif initial_parameter:
            self.weight_w = initial_parameter["weight_w"]
            self.weight_v = initial_parameter["weight_v"]
            self.bias_c = initial_parameter["bias_c"]
            self.bias_b = initial_parameter["bias_b"]
        else:
            self.bias_b = np.zeros(num_class)
            self.weight_w = np.zeros((num_class, num_hidden))
            self.bias_c = np.zeros(num_hidden)
            self.weight_v = np.zeros((num_visible, num_hidden))

class DRBM:

    def __init__(self, num_visible, num_hidden, num_class, div_num, initial_parameter=None):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.num_class = num_class
        self.div_num = div_num

        self.grad = parameters(num_visible, num_hidden, num_class, randominit=False)
        if initial_parameter:
            self.para = parameters(num_visible, num_hidden, num_class, initial_parameter=initial_parameter)
        else:
            self.para = parameters(num_visible, num_hidden, num_class)

        # div_numが1のときは従来のDRBM
        if self.div_num == 1:
            self._log1p_funclist = [self._log1p_exp_high, self._log1p_exp_low]
            self.vmarginal_prob = self._log1p_exp
            self.vmarginal = self._sigmoid
        # 0のときはDRBM(∞)
        elif self.div_num == 0:
            self.vmarginal_prob = self._marginal_inf_prob
            self.vmarginal = self._marginal_inf
        else:
            self._div_factor = 2.0 / (self.div_num-1)
            self.vmarginal_prob = self._marginal_prob
            self.vmarginal = self._marginal

        self.resume = None

    # 以下は周辺化の関数 匿名関数にするとプロセス分けできないのでこうなっている
    def _sigmoid(self, x):
        return 1/(1+np.exp(-x))
    def _log1p_exp(self, x):
        return np.piecewise(x, [x>0], self._log1p_funclist)
    def _log1p_exp_high(self, x):
        return x+np.log1p(np.exp(-x))
    def _log1p_exp_low(self, x):
        return np.log1p( np.exp(x))
    
    def _marginal(self, x):
        return np.piecewise(x, [x!=0], [self._marginal_nzero, 0])
    def _marginal_nzero(self, x):
        return -1 + self._div_factor * ( self._minus_sigmoid(self._div_factor * x) - self._minus_sigmoid( self._div_factor * self.div_num * x ) * self.div_num )
    def _marginal_prob(self, x):
        return np.piecewise(x, [x!=0], [self._marginal_prob_nzero, np.log(self.div_num)])
    def _marginal_prob_nzero(self,x):
        return -x + np.log( (1-np.exp(self._div_factor * self.div_num * x)) / (1-np.exp(self._div_factor * x)) )
    def _minus_sigmoid(self, x):
        return np.piecewise(x, [x>0], [lambda x: 1 / (np.exp(-x)-1), lambda x: -1/(np.exp(x)-1)-1])
    
    def _marginal_inf(self, x):
        return (1 / np.tanh(x)) - (1/x)
    def _marginal_inf_prob(self, x):
        return np.piecewise(np.fabs(x), [x!=0], [self._marginal_inf_prob_nzero, np.log(2)])
    def _marginal_inf_prob_nzero(self, x):
        return x+np.log( (1-np.exp(-2*x))/x )

    # 全クラス分/全データ分のA_jを計算(N,K,m)のサイズ
    def _matrix_ok_A(self, input_vector):
        return self.para.bias_c + self.para.weight_w + np.dot(input_vector, self.para.weight_v)[:, np.newaxis, :]

    # データに対してのA_j (N,m)
    def _matrix_A(self, answer, data):
        return self.para.bias_c + np.dot(answer, self.para.weight_w) + np.dot(data, self.para.weight_v)

    # N個のデータに対して(N,K)の確率の行列を返す
    def probability(self, A_matrix, normalize=True):
        energies = self.para.bias_b + np.sum( self.vmarginal_prob(A_matrix) , axis=2)

        if normalize:
            # それぞれのクラスの中から一番大きいものを引く
            max_energy = np.max(energies, axis=1)[:, np.newaxis]
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
    
    def train(self, training, test, learning_time, batch_size, optimizer, test_interval=100, dump_parameter=False, calc_train_correct_rate=False):
        if not (self.num_visible == len(training.data[0])):
            print(len(training.data[0]))
            raise TypeError
        
        resume_time = 0

        def train_correct_rate(train):
            correct_rate, correct = self.test_error(train)
            logging.info("️train correct rate: {} ({} / {})".format( correct_rate, correct, len(train.data) ))

        def test_correct_rate(test):
            correct_rate, correct = self.test_error(test)
            logging.info("️test correct rate: {} ({} / {})".format( correct_rate, correct, len(test.data) ))

        if not self.resume == None:
            resume_time = self.resume[0]
            learning_time = self.resume[1]

        logging.info("calculating initial correct rate.")
        test_correct_rate(test)

        if calc_train_correct_rate:
            train_correct_rate(training)

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
                self.grad.bias_b = q[0].get()
                self.grad.weight_w = q[0].get()
                self.grad.bias_c = q[1].get()
                self.grad.weight_v = q[1].get()
                for p in processes:
                    p.join()

                diff = optimizer.update(self.grad)
                self.para.bias_b   += diff.bias_b
                self.para.weight_w += diff.weight_w
                self.para.bias_c   += diff.bias_c
                self.para.weight_v += diff.weight_v

                if lt % test_interval == 0 and lt!=0:
                    test_correct_rate(test)
                    if calc_train_correct_rate:
                        train_correct_rate(training)

                    if dump_parameter:
                        self.save("%d_of_%d.json"%(lt,learning_time), [lt, learning_time])
                        logging.info("parameters are dumped.")

                logging.info("️training is processing. complete : {} / {}".format(lt+1, learning_time))

            except ValueError as e:
                logging.error(e)
                logging.error(traceback.format_exc())
                self.save("value-error.json")
                logging.error("error parameter saved.")
                exit()
            
            except FloatingPointError as e:
                logging.error(e)
                logging.error(traceback.format_exc())
                self.save("floating-point-error.json")
                logging.error("error parameter saved.")
                exit()

            except KeyboardInterrupt as e:
                logging.info("train interrupted.")
                exit()

        logging.info("calculating final correct rate.")
        test_correct_rate(test)
        if calc_train_correct_rate:
            train_correct_rate(training)
    
    def classify(self, input_data):
        probs = self.probability(self._matrix_ok_A(input_data), normalize=False)
        return np.argmax(probs, axis=1)
    
    def test_error(self, test):
        correct = 0
        logging.info("correct rate calculating.")
        split_num = np.ceil( len(test.answer)/10000 )
        splitted_datas = np.array_split(test.data, split_num, axis=0)
        splitted_answers = np.array_split(test.answer, split_num, axis=0)
        for d,a in zip(splitted_datas, splitted_answers):
            answers = np.dot( a, np.arange(self.num_class) )
            classified_data = np.where( answers == self.classify( d ), 1, 0)
            correct += np.sum( classified_data )
        correct_rate = correct / float(len(test.data))
        return correct_rate, correct

    def save(self, filename, training_progress=None):
        params = {
            "training_progress":training_progress,
            "num_class":self.num_class,
            "num_hidden":self.num_hidden,
            "num_visible":self.num_visible,
            "bias_b":self.para.bias_b.tolist(),
            "bias_c":self.para.bias_c.tolist(),
            "weight_w":self.para.weight_w.tolist(),
            "weight_v":self.para.weight_v.tolist(),
            "div_num":self.div_num,
        }
        if not training_progress == None:
            params["training_progress"] = training_progress
        json.dump(params, open(filename, "w+"))
    
    @staticmethod
    def load_from_json(filename):
        params = json.load(open(filename, "r"))
        drbm = DRBM(params["num_visible"], params["num_hidden"], params["num_class"], params["div_num"])
        drbm.para.bias_b = np.array(params["bias_b"])
        drbm.para.bias_c = np.array(params["bias_c"])
        drbm.para.weight_w = np.array(params["weight_w"])
        drbm.para.weight_v = np.array(params["weight_v"])
        drbm.resume = params["training_progress"]
        return drbm

def main():
    drbm = DRBM(3, 5, 7)
    print(a)
    print(sum(a))

if __name__=='__main__':
    main()