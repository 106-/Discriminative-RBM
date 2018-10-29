#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

np.seterr(all="raise")

div = 2
factor = 2.0 / (div-1)

def marginal_prob(x):
    return np.piecewise(x,
        [
            x < -1e-3,
            1e-3 < x
        ], 
        [
            lambda x: -x + np.log( (1-np.exp(factor * div * x)) / (1-np.exp(factor * x)) ),
            lambda x: -x + np.log( (np.exp(- factor * div * x)-1)/(np.exp(- factor * div * x) - np.exp(factor * x * (1-div))) ), 
            lambda x: -x + np.log( div + 0.5 * factor * (div-1) * div * x
                        + (1/12) * factor**2 * div * (2 * div**2 - 3 * div + 1) * x**2
                        + (1/24) * factor**3 * (div-1)**2 * div**2 * x**3 )
        ]
    )


a = np.linspace(-10, 10, 100000)
b = marginal_prob(a)

plt.plot(a,b,label="2")
plt.legend()
plt.show()