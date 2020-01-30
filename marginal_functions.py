
import numpy as np
import numexpr as ne
import logging

# act は左右対称(あるいはsoftplus)のやつ
# diff はtanhみたいなやつ

def softplus(x):
    return np.piecewise(x, [x<0], [
        lambda x: ne.evaluate("log(1+exp(x))"),
        lambda x: ne.evaluate("x+log(1+exp(-x))")
    ])

class original:
    def act(self, x):
        return softplus(x)

    def diff(self, x):
        return ne.evaluate("1/(1+exp(-x))")

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
            lambda x: ne.evaluate("1 / (exp(-x)-1)"),
            lambda x: ne.evaluate("-1 / (exp(x)-1)-1")
        ])

class multiple_continuous:
    def act(self, x):
        return np.piecewise(np.fabs(x), [x==0], [
            np.log(2),
            lambda x: ne.evaluate("x+log( (1-exp(-2*x))/x )")
        ])

    def diff(self, x):
        return np.piecewise(x, [x==0], [
            0,
            lambda x: ne.evaluate("(1 / tanh(x)) - (1/x)")
        ])

class sparse_continuous:
    def __init__(self, hidden_num, lambda_vector=None, initial_lambda=10.0, use_adamax=False, learning_rate=1.0):
        if lambda_vector is None:
            self.lambda_vector = np.full(hidden_num, initial_lambda)
        else:
            self.lambda_vector = lambda_vector
        
        self.use_adamax = use_adamax
        if use_adamax:
            self._alpha = 0.002
            self._beta1 = 0.9
            self._beta2 = 0.999
            self._epsilon = 1e-8
            self._t = 0

            self._moment = np.zeros(self.lambda_vector.shape)
            self._norm = np.zeros(self.lambda_vector.shape)
            self._diff = np.zeros(self.lambda_vector.shape)
            logging.info("sparse param updating method: Adamax")
        else:
            logging.info("sparse param updating method: SGD({})".format(learning_rate))
            self._learning_rate = learning_rate
    
    def fit_lambda(self, a_data, a_ok, probs_matrix):
        grad_lambda = np.mean( self.ldiff(a_data) - np.sum( self.ldiff(a_ok) * probs_matrix[:, :, np.newaxis], axis=1 ), axis=0 )

        if self.use_adamax:
            # Adamax
            self._t += 1
            self._moment = self._moment * self._beta1 + grad_lambda * (1-self._beta1)
            self._norm = np.max( np.vstack((self._norm * self._beta2, np.abs(grad_lambda))), axis=0)
            diff = self._moment / (self._norm + self._epsilon) * (self._alpha/(1-np.power(self._beta1, self._t)))
            np.sum((self.lambda_vector, diff), axis=0, out=self.lambda_vector)
        else:
            np.sum((self.lambda_vector, grad_lambda * self._learning_rate), axis=0, out=self.lambda_vector)

    def _separation(self, x):
        b = softplus(self.lambda_vector)
        return ne.evaluate("(b+x)/2"), ne.evaluate("(b-x)/2")

    def act(self, x):
        a = self.a(x)
        return ne.evaluate("log(a)")

    def a(self, x):
        return self._sp_a(*self._separation(x))
    
    def _sp_a(self, a, b):
        return self._s(a) + self._s(b)

    def diff(self, x):
        a, b = self._separation(x)
        spa = self._sp_a(a, b)
        sa = self._s_grad(a)
        sb = self._s_grad(b)
        return ne.evaluate("1 / (2*spa) * (sa-sb)")
    
    def ldiff(self, x):
        a, b = self._separation(x)
        spa = self._sp_a(a, b)
        sa = self._s_grad(a)
        sb = self._s_grad(b)
        return ne.evaluate("1 / (2*spa) * (sa+sb)")

    def _s(self, x):
        return np.piecewise(x, [x==0, np.abs(x)<1e-4], [
            1,
            lambda x: ne.evaluate("1 - x + 2*x**2/3 - x**3/3 + 2*x**4/15")
            lambda x: ne.evaluate("exp(-x) * sinh(x) / x")
        ])
    
    def _s_grad(self, x):
        return np.piecewise(x, [x==0, np.abs(x)<1e-4], [
            -1,
            lambda x: ne.evaluate("-1 + 4*x/3 - x**2 + 8*x**3/15 - 2*x**4/9")
            lambda x: ne.evaluate("exp(-x) * sinh(x) / x * (1/tanh(x)-1/x-1)")
        ])
