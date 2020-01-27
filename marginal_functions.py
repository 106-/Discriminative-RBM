
import numpy as np
import numexpr as ne

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
        else:
            self._learning_rate = learning_rate
    
    def act(self, x):
        a, b = self._get_separation_calc(x)
        return self._J(b, -a)
    
    def diff(self, x, **kwargs):
        a, b = self._get_separation_calc(x)
        return self._U(b, -a)
    
    def fit_lambda(self, a_data, a_ok, probs_matrix):
        a, b = self._get_separation_calc(a_data)
        mar_a_data = self._U(b, -a, alpha=-0.5)
        a, b = self._get_separation_calc(a_ok)
        mar_a_ok = self._U(b, -a, alpha=-0.5)
        # diff_softplus -> (m)
        diff_softplus = -1 / (1 + np.exp(-self.lambda_vector) )

        sum_k = np.sum( mar_a_ok * probs_matrix[:, :, np.newaxis], axis=1)
        # diff_mar_a -> (N,m)
        diff_mar_a = mar_a_data - sum_k
        # (m) * (N,m) なのでbroadcastが有効
        grad_lambda = diff_softplus * np.sum( diff_mar_a, axis=0) / len(a_data)

        if self.use_adamax:
            # Adamax
            self._t += 1
            self._moment = self._moment * self._beta1 + grad_lambda * (1-self._beta1)
            self._norm = np.max( np.vstack((self._norm * self._beta2, np.abs(grad_lambda))), axis=0)
            diff = self._moment / (self._norm + self._epsilon) * (self._alpha/(1-np.power(self._beta1, self._t)))
            np.sum((self.lambda_vector, diff), axis=0, out=self.lambda_vector)
        else:
            np.sum((self.lambda_vector, grad_lambda * self._learning_rate), axis=0, out=self.lambda_vector)

    def _get_separation_calc(self, x):
        sp_lambda = softplus(self.lambda_vector)
        a = ne.evaluate("(x - sp_lambda)/2")
        b = ne.evaluate("(x + sp_lambda)/2")
        return (a, b)

    def _Q(self, b, a):
        return self._K(b) / self._K(a)

    def _J(self, a, b):
        higher_than_zero_a = 0<=a
        return np.piecewise(a, [ higher_than_zero_a ], [ 
            lambda a: np.log(self._K(a)) + np.log1p(self._Q(b[higher_than_zero_a], a)),
            lambda a: np.log(self._K(-a)) -2*a + np.log1p(self._Q(b[np.logical_not(higher_than_zero_a)], a))
        ])

    def _K(self, x):
        return np.piecewise(x, [ x<-1e-3, 1e-3<x ], [
            lambda x: ne.evaluate("exp(-x)*sinh(abs(x))/abs(x)"),
            lambda x: ne.evaluate("1/(2*x)*(1-exp(-2*x))"),
            lambda x: ne.evaluate("1 - x + 2.0/3.0*x**2 - 1.0/3.0*x**3")
        ])

    def _U(self, a,b, alpha=0.5, beta=-0.5):
        return ( alpha * self._r(a) + beta * self._r(b) * self._Q(b,a) ) / (1+self._Q(b,a))

    def _r(self, x):
        return np.piecewise(x, [ np.abs(x) < 1e-3 ], [
            lambda x: ne.evaluate("-1 + x/3 - x**3/45"),
            lambda x: ne.evaluate("1/tanh(x) - 1/x -1")
        ])

if __name__ == '__main__':
    N = 100
    m = 10
    K = 10
    probs = np.random.rand(N, K)
    a_data = np.random.rand(N, m)
    a_ok = np.random.rand(N, K, m)
    s = sparse_continuous(10)
    s.fit_lambda(a_data, a_ok, probs)