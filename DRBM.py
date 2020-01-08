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
from marginal_functions import *
import multiprocessing as mp


class parameters:
    def __init__(self, num_visible, num_hidden, num_class, randominit=True, initial_parameter=None, random_bias=False):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.num_class = num_class

        if randominit and not initial_parameter :
            # Xavierの初期値
            uniform_range = np.sqrt( 6/(num_visible + num_hidden) )
            self.weight_v = np.random.uniform(-uniform_range, uniform_range, (num_visible, num_hidden))
            uniform_range = np.sqrt( 6/(num_hidden + num_class) )
            self.weight_w = np.random.uniform(-uniform_range, uniform_range, (num_class, num_hidden))
            if not random_bias:
                # バイアスは0で初期化
                self.bias_c = np.zeros(num_hidden)
                self.bias_b = np.zeros(num_class)
            else:
                self.bias_c = np.random.randn(num_hidden)
                self.bias_b = np.random.randn(num_class)
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
    
    def _arithmetic(self, other, operation_func):
        res = parameters(self.num_visible, self.num_hidden, self.num_class, randominit=False)
        if isinstance(other, parameters):
            operation_func(self.bias_b, other.bias_b, out=res.bias_b)
            operation_func(self.bias_c, other.bias_c, out=res.bias_c)
            operation_func(self.weight_v, other.weight_v, out=res.weight_v)
            operation_func(self.weight_w, other.weight_w, out=res.weight_w)
        # 演算の対象がparamtersでない場合は,そのままの計算を試みる
        else:
            operation_func(self.bias_b, other, out=res.bias_b)
            operation_func(self.bias_c, other, out=res.bias_c)
            operation_func(self.weight_v, other, out=res.weight_v)
            operation_func(self.weight_w, other, out=res.weight_w)
        return res

    def __add__(self, other):
        return self._arithmetic(other, np.add)
    def __mul__(self, other):
        return self._arithmetic(other, np.multiply)
    def __truediv__(self, other):
        return self._arithmetic(other, np.true_divide)
    def __sub__(self, other):
        return self._arithmetic(other, np.subtract)
    def __pow__(self, other):
        return self._arithmetic(other, np.power)

    def map(self, npfunc):
        res = parameters(self.num_visible, self.num_hidden, self.num_class, randominit=False)
        npfunc(self.bias_b, out=res.bias_b)
        npfunc(self.bias_c, out=res.bias_c)
        npfunc(self.weight_v, out=res.weight_v)
        npfunc(self.weight_w, out=res.weight_w)
        return res

    def __abs__(self):
        return self.map(np.fabs)
    
    # 2つのパラメータを比較して,大きいほうのみを抽出する関数
    def max(self, para_b):
        res = parameters(self.num_visible, self.num_hidden, self.num_class, randominit=False)
        np.max( np.concatenate((self.bias_b[np.newaxis, :], para_b.bias_b[np.newaxis, :]), axis=0), axis=0, out=res.bias_b )
        np.max( np.concatenate((self.bias_c[np.newaxis, :], para_b.bias_c[np.newaxis, :]), axis=0), axis=0, out=res.bias_c )
        np.max( np.concatenate((self.weight_w[np.newaxis, :, :], para_b.weight_w[np.newaxis, :, :]), axis=0), axis=0, out=res.weight_w )
        np.max( np.concatenate((self.weight_v[np.newaxis, :, :], para_b.weight_v[np.newaxis, :, :]), axis=0), axis=0, out=res.weight_v )
        return res

class DRBM:

    def __init__(self, num_visible, num_hidden, num_class, div_num, initial_parameter=None, enable_sparse=False, random_bias=False):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.num_class = num_class
        self.div_num = div_num
        self.enable_sparse = enable_sparse

        self.grad = parameters(num_visible, num_hidden, num_class, randominit=False)
        if initial_parameter:
            self.para = initial_parameter
        else:
            self.para = parameters(num_visible, num_hidden, num_class, random_bias=random_bias)

        if self.enable_sparse:
            logging.info("enable sparse normalization for hidden layer.")
            self.marginal = sparse_continuous(num_hidden)
        # div_numが1のときは従来のDRBM
        elif self.div_num == 1:
            self.marginal = original()
        # 0のときはDRBM(∞)
        elif self.div_num == 0:
            self.marginal = multiple_continuous()
        else:
            self.marginal = multiple_discrete(div_num)

        self.resume = None

    # 全クラス分/全データ分のA_jを計算(N,K,m)のサイズ
    def matrix_ok_A(self, input_vector):
        return self.para.bias_c + self.para.weight_w + np.dot(input_vector, self.para.weight_v)[:, np.newaxis, :]

    # データに対してのA_j (N,m)
    def _matrix_A(self, answer, data):
        return self.para.bias_c + np.dot(answer, self.para.weight_w) + np.dot(data, self.para.weight_v)

    # N個のデータに対して(N,K)の確率の行列を返す
    def probability(self, A_matrix, normalize=True):
        energies = self.para.bias_b + np.sum( self.marginal.act(A_matrix) , axis=2)

        if normalize:
            # それぞれのクラスの中から一番大きいものを引く
            max_energy = np.max(energies, axis=1)[:, np.newaxis]
            energies = np.exp(energies-max_energy)
            return energies / np.sum(energies, axis=1)[:, np.newaxis]
        else:
            return energies

    # b,wの勾配を計算diff_bは(K),diff_wは(K,m)
    def _differential_bw(self, training):
        diff_tp = training.answer - self._probs_matrix
        diff_b = np.sum(diff_tp, axis=0) / len(training.data)
        diff_w = np.sum(np.multiply(self._A_matrix_ok, diff_tp[:, :, np.newaxis]), axis=0) / len(training.data)
        return diff_b, diff_w
    
    # c,vの勾配 diff_cは(m), diff_vは(n,m)
    def _differential_cv(self, training):
        sum_under_k = np.sum( self._sig_A_ok * self._probs_matrix[:, :, np.newaxis], axis=1)
        diff_sig_a = self._sig_A - sum_under_k
        diff_c = np.sum( diff_sig_a, axis=0 ) / len(training.data)
        diff_v = np.sum( np.multiply( training.data[:, :, np.newaxis], diff_sig_a[:, np.newaxis, :]), axis=0) / len(training.data)
        return diff_c, diff_v
    
    def train(self, training, test, learning_time, batch_size, optimizer, test_interval=100, dump_parameter=False, correct_rate=False, gen_drbm=None):

        learning_result = LearningResult(learning_time, optimizer.__class__.__name__, len(training.data), len(test.data), batch_size, test_interval, self)

        if not (self.num_visible == len(training.data[0])):
            print(len(training.data[0]))
            raise TypeError
        
        def train_correct_rate(lt, train):
            correct_rate, correct = self.test_error(train)
            logging.info("️train correct rate: {} ({} / {})".format( correct_rate, correct, len(train.data)))
            learning_result.make_log(lt, "train-correct-rate", correct_rate)

        def test_correct_rate(lt, test):
            correct_rate, correct = self.test_error(test)
            logging.info("️test correct rate: {} ({} / {})".format( correct_rate, correct, len(test.data)))
            learning_result.make_log(lt, "test-correct-rate", correct_rate)

        def calc_kld(lt):
            kld_mean = self.kl_divergence(gen_drbm)
            logging.info("KL-Divergence mean: {}".format(kld_mean))
            learning_result.make_log(lt, "KL-Divergence", kld_mean)

        if correct_rate:
            logging.info("calculating initial correct rate.")
            train_correct_rate(0, training)
            test_correct_rate(0, test)

        if gen_drbm is not None:
            logging.info("calculating initial KL-Divergence.")
            calc_kld(0)

        if self.enable_sparse:
            learning_result.make_log(0, "sparse_parameter_max", np.max(self.marginal.lambda_vector).tolist())
            learning_result.make_log(0, "sparse_parameter_avg", np.average(self.marginal.lambda_vector).tolist())

        for lt in range(learning_time):
            try:
                batch = training.minibatch(batch_size)
                self._A_matrix_ok = self.matrix_ok_A(batch.data)
                self._A_matrix = self._matrix_A(batch.answer, batch.data)
                self._probs_matrix = self.probability(self._A_matrix_ok)
                self._sig_A_ok = self.marginal.diff(self._A_matrix_ok)
                self._sig_A = self.marginal.diff(self._A_matrix)

                if self.enable_sparse:
                    self.marginal.fit_lambda(self._A_matrix, self._A_matrix_ok, self._probs_matrix)
                
                self.grad.bias_b, self.grad.weight_w = self._differential_bw(batch)
                self.grad.bias_c, self.grad.weight_v = self._differential_cv(batch) 

                diff = optimizer.update(self.grad)
                self.para.bias_b   += diff.bias_b
                self.para.weight_w += diff.weight_w
                self.para.bias_c   += diff.bias_c
                self.para.weight_v += diff.weight_v

                if lt % test_interval == 0 and lt!=0:
                    if correct_rate:
                        test_correct_rate(lt, test)
                        train_correct_rate(lt, training)

                    # if gen_drbm is not None:
                    #     calc_kld(lt)

                    if dump_parameter:
                        self.save("%d_of_%d.json"%(lt,learning_time), [lt, learning_time])
                        logging.info("parameters are dumped.")

                    if self.enable_sparse:
                        learning_result.make_log(lt, "sparse_parameter_max", np.max(self.marginal.lambda_vector).tolist())
                        learning_result.make_log(lt, "sparse_parameter_avg", np.average(self.marginal.lambda_vector).tolist())

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
        if correct_rate:
            test_correct_rate(lt, test)
            train_correct_rate(lt, training)
            
        if self.enable_sparse:
            learning_result.make_log(lt, "sparse_parameter_max", np.max(self.marginal.lambda_vector).tolist())
            learning_result.make_log(lt, "sparse_parameter_avg", np.average(self.marginal.lambda_vector).tolist())
            
        if gen_drbm is not None:
            calc_kld(lt)
    
        return learning_result
    
    def classify(self, input_data):
        probs = self.probability(self.matrix_ok_A(input_data), normalize=False)
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

    def kl_divergence(self, gen_drbm, sampling_num=1000):
        logging.info("calclating KL-Divergence.")
        random_value, gen_probs = gen_drbm.sampling(sampling_num)

        a_matrix = self.matrix_ok_A(random_value)
        probs = self.probability(a_matrix)

        kld = np.sum( gen_probs * np.log( gen_probs / probs ), axis=1 )
        kld_mean = np.mean(kld)
        return kld_mean

    def sampling(self, sampling_num):
        random_value = np.random.randn(sampling_num, self.num_visible)
        a_matrix = self.matrix_ok_A(random_value)
        probs = self.probability(a_matrix)
        return random_value, probs

    def save(self, filename, training_progress=None):
        params = {
            "num_class":self.num_class,
            "num_hidden":self.num_hidden,
            "num_visible":self.num_visible,
            "params":{
                "bias_b":self.para.bias_b.tolist(),
                "bias_c":self.para.bias_c.tolist(),
                "weight_w":self.para.weight_w.tolist(),
                "weight_v":self.para.weight_v.tolist(),
            }
        }
        if self.enable_sparse:
            params["sparse_params"] = self.marginal.lambda_vector.tolist()
        json.dump(params, open(filename, "w+"), indent=2)
    
    @staticmethod
    def load_from_json(filename, hidden_division=2, enable_sparse=False):
        data = json.load(open(filename, "r"))
        drbm = DRBM(data["num_visible"], data["num_hidden"], data["num_class"], hidden_division, enable_sparse=enable_sparse)
        for p in data["params"]:
            setattr(drbm.para, p, np.array(data["params"][p]))
        if enable_sparse and "sparse_params" in data:
            drbm.marginal.lambda_vector = np.array(data["sparse_params"])
        return drbm

class LearningResult:
    def __init__(self, learning_num, optimize_method, train_data_length, test_data_length, batch_size, test_interval, drbm):
        self.testament = {
            "learning_num" : learning_num,
            "optimize_method" : optimize_method,
            "train_data_length" : train_data_length,
            "test_data_length" : test_data_length,
            "batch_size" : batch_size,
            "test_interval" : test_interval,
            "DRBM" : {
                "num_class":drbm.num_class,
                "num_hidden":drbm.num_hidden,
                "num_visible":drbm.num_visible,
                # "bias_b":drbm.para.bias_b.tolist(),
                # "bias_c":drbm.para.bias_c.tolist(),
                # "weight_w":drbm.para.weight_w.tolist(),
                # "weight_v":drbm.para.weight_v.tolist(),
                "div_num":drbm.div_num,
                "enable_sparse": drbm.enable_sparse,
            },
            "log":{}
        }
    
    def make_log(self, learning_count, value_name, value):
        if not value_name in self.testament["log"]:
            self.testament["log"][value_name] = []
        self.testament["log"][value_name].append( [learning_count, value] )
    
    def save(self, filename):
        json.dump(self.testament, open(filename, "w+"))