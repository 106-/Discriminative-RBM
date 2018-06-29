#!/usr/bin/env python
# -*- coding:utf-8 -*-

import math
import random

class DRBM:

    def __init__(self, num_visible, num_hidden, num_class):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.num_class = num_class
        # Xavierの初期値
        sqrt_max_num_of_node = math.sqrt(max(num_visible, num_hidden, num_class))
        self.weight_v = [[random.gauss(0, 1/sqrt_max_num_of_node) for j in range(num_hidden)] for i in range(num_visible)]
        self.weight_w = [[random.gauss(0, 1/sqrt_max_num_of_node) for k in range(num_class)] for j in range(num_hidden)]
        self.bias_c = [random.gauss(0, 1/sqrt_max_num_of_node) for j in range(num_hidden)]
        self.bias_b = [random.gauss(0, 1/sqrt_max_num_of_node) for k in range(num_class)]

    @staticmetho
    def _sigmoid(x):
        return 1/(1+math.exp(-x))

    # log(1+e^x)を安全に計算する
    @staticmethod
    def _log_exp(x):
        if x<0:
            return math.exp(x)
        else:
            return (x+math.log(1+math.exp(-x)))

    # one-of-k表現のベクトルを生成
    @staticmethod
    def one_of_k(size, number):
        # "One of K" -> "ok"
        ok = [0 for i in range(size)]
        ok[number] = 1
        return ok

    # 31PにあるA_jを計算
    def _calc_A(self, input_vector, one_of_k, j=None, update=False):
        def func(j):
            weight_v_under_j = list(map(lambda x:x[j], self.weight_v))
            weight_w_under_j = self.weight_w[j]
            sum_wt = 0
            for w,t in zip(weight_w_under_j, one_of_k):
                sum_wt += w*t
            sum_vx = 0
            for v,x in zip(weight_v_under_j, input_vector):
                sum_vx += v*x
            return self.bias_c[j] + sum_wt + sum_vx
    
        a = None

        if update == True:
            # jがNoneなら全てのA_jを計算
            if j == None:
                a = []
                for j in range(self.num_hidden):
                    a.append(func(j))
                self._old_a = a
            # jが整数ならその値に対するA_jを計算
            elif isinstance(j, int):
                a = func(j)
                self._old_a[j] = a
        else:
            if j == None:
                a = self._old_a
            elif isinstance(j, int):
                a = self._old_a[j]
            
        return a

    def probability(self, input_vector, k=None, update=False):
        if update == True:
            sum_exp_energies = []
            for k_c in range(self.num_class):
                A = self._calc_A(input_vector, self.one_of_k(self.num_class, k_c), update=True)
                sum_ln = sum(list(map(lambda x: 1+math.exp(x), A)))
                sum_exp_energies.append(math.exp(self.bias_b[k_c]) + sum_ln)
            # nc -> Normalize Constant
            nc = sum(sum_exp_energies)
            probs = list(map(lambda x: x/nc, sum_exp_energies))

            if k==None:
                self._old_prob = probs
                return probs
            elif isinstance(k, int):
                self._old_prob[k] = probs[k]
                return probs[k]
        else:
            if k == None:
                return self._old_prob
            elif isinstance(k, int):
                return self._old_prob[k]

    def _differential_b(self, training_datas, training_answers):
        diff_b = []
        for k in range(self.num_class):
            sum_differ = 0
            for t_u, x_u in zip(training_answers, training_datas):
                sum_differ += t_u[k] - self.probability(x_u, k=k)
            diff_b.append(sum_differ / len(training_datas))
        return diff_b
    
    def _differential_w(self, training_datas, training_answers):
        diff_w = [[None for k in range(self.num_class)] for j in range(self.num_hidden)]
        for j in range(self.num_hidden):
            for k in range(self.num_class):
                sum_sig = 0
                for t_u, x_u in zip(training_answers, training_datas):
                    sum_sig += self._sigmoid(self._calc_A(x_u, self.one_of_k(self.num_class, k), j=j)) * (t_u[k] - self.probability(x_u, k=k))
                diff_w[j][k] = sum_sig / len(training_datas)
        return diff_w
    
    def _differential_c(self, training_datas, training_answers):
        diff_c = []
        for j in range(self.num_hidden):
            sum_ = 0
            for t_u, x_u in zip(training_answers, training_datas):
                sum_data_sig = self._sigmoid(self._calc_A(x_u, t_u, j=j))
                sum_sig = 0
                for k in range(self.num_class):
                    sum_sig += self._sigmoid(self._calc_A(x_u, self.one_of_k(self.num_class, k), j=j)) * self.probability(x_u, k=k)
                sum_ += (sum_data_sig - sum_sig)
            diff_c.append(sum_ / len(training_datas))
        return diff_c
    
    def _differential_v(self, training_datas, training_answers):
        diff_v = [[None for j in range(self.num_hidden)] for i in range(self.num_visible)]
        for i in range(self.num_visible):
            for j in range(self.num_hidden):
                sum_ = 0
                for t_u, x_u in zip(training_answers, training_datas):
                    sig = self._sigmoid(self._calc_A(x_u, t_u, j=j))
                    sum_sig_p = 0
                    for k in range(self.num_class):
                        sum_sig_p += self._sigmoid(self._calc_A(x_u, self.one_of_k(self.num_class, k), j=j)) * self.probability(x_u, k=k)
                    sum_ += x_u[i] * (sig - sum_sig_p)
                diff_v[i][j] = sum_ / len(training_datas)
        return diff_v            
                
    def train(self, training_datas, training_answers, test_datas, test_answers, learning_time, learning_rate=[0.01, 0.01, 0.01, 0.01], test_interval=100):
        if not (self.num_visible == len(training_datas[0])):
            print(len(training_datas[0]))
            raise TypeError
        
        for lt in range(learning_time):
            self.probability(training_datas[0], update=True)

            diff_b = self._differential_b(training_datas, training_answers)
            diff_w = self._differential_w(training_datas, training_answers)
            diff_c = self._differential_c(training_datas, training_answers)
            diff_v = self._differential_v(training_datas, training_answers)

            for k in range(self.num_class):
                self.bias_b[k] += diff_b[k] * learning_rate[0]
            
            for j in range(self.num_hidden):
                for k in range(self.num_class):
                    self.weight_w[j][k] += diff_w[j][k] * learning_rate[1]
            
            for j in range(self.num_hidden):
                self.bias_c[j] += diff_c[j] * learning_rate[2]
            
            for i in range(self.num_visible):
                for j in range(self.num_hidden):
                    self.weight_v[i][j] += diff_v[i][j] * learning_rate[3]
            
            if lt % test_interval == 0:
                print("{} / {}".format(lt, len(training_answers)))
                self.test(test_datas, test_answers)
    
    def classify(self, input_data):
        probs = self.probability(input_data)
        return probs.index(max(probs))
    
    def test(self, test_datas, test_answers):
        correct = 0
        for d, a in zip(test_datas, test_answers):
            if a == self.classify(d):
                correct += 1
        print("correct rate: {}".format(correct / len(test_datas)))

def main():
    drbm = DRBM(3, 5, 2)
    print(drbm.probability([123,56,23]))

if __name__=='__main__':
    main()