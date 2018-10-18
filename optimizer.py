# -*- coding:utf-8 -*-
 
import numpy as np
from DRBM import parameters

class optimizer:
    def __init__(self):
        pass
    
    def update(self):
        pass

class momentum(optimizer):
    # 更新レートはb,w,c,vの順番
    def __init__(self, num_visible, num_hidden, num_class, learning_rate=[0.01, 0.01, 0.1, 0.1], alpha=[0.9, 0.9, 0.9, 0.9]):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.num_class = num_class
        self.learning_rate = learning_rate
        self.alpha = alpha

        self._old_grad = parameters(num_visible, num_hidden, num_class, randominit=False)
        self._diff = parameters(num_visible, num_hidden, num_class, randominit=False)
    
    def update(self, grad):
        self._diff.bias_b = grad.bias_b * self.learning_rate[0] + self._old_grad.bias_b * self.alpha[0]
        self._diff.weight_w = grad.weight_w * self.learning_rate[1] + self._old_grad.weight_w * self.alpha[1]
        self._diff.bias_c = grad.bias_c * self.learning_rate[2] + self._old_grad.bias_c * self.alpha[2]
        self._diff.weight_v = grad.weight_v * self.learning_rate[3] + self._old_grad.weight_v * self.alpha[3]
        self._old_grad = grad
        return self._diff
