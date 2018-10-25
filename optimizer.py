# -*- coding:utf-8 -*-
 
import numpy as np
from DRBM import parameters
import logging

class optimizer:
    def __init__(self):
        pass
    
    def update(self):
        pass

class momentum(optimizer):
    # 更新レートはb,w,c,vの順番
    def __init__(self, num_visible, num_hidden, num_class, learning_rate=[0.01, 0.01, 0.1, 0.1], alpha=[0.9, 0.9, 0.9, 0.9]):
        self._learning_rate = learning_rate
        self._alpha = alpha

        self._old_grad = parameters(num_visible, num_hidden, num_class, randominit=False)
        self._diff = parameters(num_visible, num_hidden, num_class, randominit=False)
    
    def update(self, grad):
        self._diff.bias_b = grad.bias_b * self._learning_rate[0] + self._old_grad.bias_b * self._alpha[0]
        self._diff.weight_w = grad.weight_w * self._learning_rate[1] + self._old_grad.weight_w * self._alpha[1]
        self._diff.bias_c = grad.bias_c * self._learning_rate[2] + self._old_grad.bias_c * self._alpha[2]
        self._diff.weight_v = grad.weight_v * self._learning_rate[3] + self._old_grad.weight_v * self._alpha[3]
        self._old_grad = grad
        return self._diff

class adamax(optimizer):
    def __init__(self, num_visible, num_hidden, num_class, alpha=0.002, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self._alpha = alpha
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._t = 0

        self._moment = parameters(num_visible, num_hidden, num_class, randominit=False)
        self._norm = parameters(num_visible, num_hidden, num_class, randominit=False)
        self._diff = parameters(num_visible, num_hidden, num_class, randominit=False)
    
    def update(self, grad):
        self._t += 1
        self._moment = self._moment * self._beta1 + grad * (1-self._beta1)
        self._norm = parameters.max( self._norm * self._beta2, abs(grad) )
        self._diff = self._moment / (self._norm + self._epsilon) * (self._alpha/(1-np.power(self._beta1, self._t)))
        return self._diff

class adam(optimizer):
    def __init__(self, num_visible, num_hidden, num_class, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self._alpha = alpha
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._t = 0

        self._moment_m = parameters(num_visible, num_hidden, num_class, randominit=False)
        self._moment_v = parameters(num_visible, num_hidden, num_class, randominit=False)

    def update(self, grad):
        self._t += 1
        self._moment_m = self._moment_m * self._beta1 + grad * (1-self._beta1)
        self._moment_v = self._moment_v * self._beta2 + grad**2 * (1-self._beta2)
        m_hat = self._moment_m / (1-np.power(self._beta1, self._t))
        v_hat = self._moment_v / (1-np.power(self._beta2, self._t))
        diff = m_hat / ( v_hat.map(np.sqrt) + self._epsilon ) * self._alpha
        return diff
