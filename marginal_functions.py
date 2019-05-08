
import numpy as np

# act は左右対称(あるいはsoftplus)のやつ
# diff はtanhみたいなやつ

def softplus(x):
    return np.piecewise(x, [x<0], [
        lambda x: np.log(1+np.exp(x)),
        lambda x: x+np.log(1+np.exp(-x))
    ])

class original:
    def act(self, x):
        return softplus(x)

    def diff(self, x):
        return 1/(1+np.exp(-x))

class multiple_discrete:
    def __init__(self, div_num):
        self._div_num = div_num
        self._div_factor = 2.0 / (self._div_num-1)
    
    def act(self, x):
        return np.piecewise(x, [ x < -1e-3, 1e-3 < x ], [
                lambda x: -x + np.log( (1-np.exp(self._div_factor * self._div_num * x)) / (1-np.exp(self._div_factor * x)) ),
                lambda x: -x + np.log( (np.exp(- self._div_factor * self._div_num * x)-1)/(np.exp(- self._div_factor * self._div_num * x) - np.exp(self._div_factor * x * (1-self._div_num))) ), 
                lambda x: -x + np.log( self._div_num + 0.5 * self._div_factor * (self._div_num-1) * self._div_num * x
                            + (1/12) * self._div_factor**2 * self._div_num * (2 * self._div_num**2 - 3 * self._div_num + 1) * x**2
                            + (1/24) * self._div_factor**3 * (self._div_num-1)**2 * self._div_num**2 * x**3)      
        ])
    
    def diff(self, x):
        return np.piecewise(x, [x==0], [
            0,
            lambda x: -1 + self._div_factor * ( self._minus_sigmoid(self._div_factor * x) - self._minus_sigmoid( self._div_factor * self._div_num * x ) * self._div_num )
        ])
    
    def _minus_sigmoid(self, x):
        return np.piecewise(x, [x>0], [
            lambda x: 1 / (np.exp(-x)-1),
            lambda x: -1 / (np.exp(x)-1)-1
        ])

class multiple_continuous:
    def act(self, x):
        return np.piecewise(np.fabs(x), [x==0], [
            np.log(2),
            lambda x: x+np.log( (1-np.exp(-2*x))/x )
        ])

    def diff(self, x):
        return np.piecewise(x, [x==0], [
            0,
            lambda x: (1 / np.tanh(x)) - (1/x)
        ])