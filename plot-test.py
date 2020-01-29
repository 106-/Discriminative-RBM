#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

def marginal_sparse(x, term_num, lambda_vector):
    splambda = soft_plus(lambda_vector)

    bottom_x = x + splambda
    top_x = x - splambda

    bottom_term_num = np.ceil(term_num / 2)
    top_term_num = np.floor(term_num / 2)
    ratio = 2 / (term_num-1)
    
    higher_zero = x>0
    return np.piecewise(x, [higher_zero], [
        lambda x: np.log( np.exp(-2*x) * sum_geo(bottom_x[higher_zero], ratio, bottom_term_num) + sum_geo(top_x[higher_zero], -ratio, top_term_num) ) + x,
        lambda x: np.log( np.exp(2*x) * sum_geo(top_x[np.logical_not(higher_zero)], -ratio, top_term_num) + sum_geo(bottom_x[np.logical_not(higher_zero)], ratio, bottom_term_num) ) - x
    ]) - splambda

def sum_geo(x, ratio, term_num):
    return np.piecewise(x, [x*ratio*term_num<0], [
        lambda x: ( (1-np.exp(x*ratio*term_num))/(1-np.exp(x*ratio)) ),
        lambda x: np.exp(term_num) * (np.exp(-x*ratio*term_num)-1) / (np.exp(-x*ratio)-1)
    ])

def soft_plus(x):
    return np.piecewise(x, [x>0], [lambda x: x+np.log(np.exp(-x)+1), lambda x: np.log(1+np.exp(x))])

def marginal_prob(x, div_factor, div_num):
    return np.exp(-x) * (1-np.exp(div_factor * div_num * x)) / (1-np.exp(div_factor * x)) 
    #return np.log( np.exp(-x) * (1-np.exp(div_factor * div_num * x)) / (1-np.exp(div_factor * x)) )

def marginal_inf(x):
    return np.log( 2 * np.sinh(x) / x )

def marginal_sparse(x, term_num, lambda_vector):
    splambda = soft_plus(lambda_vector)

    bottom_x = x + splambda
    top_x = x - splambda

    bottom_term_num = np.ceil(term_num / 2)
    top_term_num = np.floor(term_num / 2)
    ratio = 2 / (term_num-1)
    
    #return np.log( np.exp(-bottom_x)*((1-np.exp(ratio*bottom_term_num*bottom_x))/(1-np.exp(ratio*bottom_x))) 
    #        + np.exp(top_x)*((1-np.exp(-ratio*top_term_num*top_x))/(1-np.exp(-ratio*top_x))))
    return np.exp(-bottom_x)*((1-np.exp(ratio*bottom_term_num*bottom_x))/(1-np.exp(ratio*bottom_x))) + np.exp(top_x)*((1-np.exp(-ratio*top_term_num*top_x))/(1-np.exp(-ratio*top_x)))
#
#    higher_zero = x>0
#    return np.piecewise(x, [higher_zero], [
#        lambda x: np.log( np.exp(-2*x) * sum_geo(bottom_x[higher_zero], ratio, bottom_term_num) + sum_geo(top_x[higher_zero], -ratio, top_term_num) ) + x,
#        lambda x: np.log( np.exp(2*x) * sum_geo(top_x[np.logical_not(higher_zero)], -ratio, top_term_num) + sum_geo(bottom_x[np.logical_not(higher_zero)], ratio, bottom_term_num) ) - x
#    ]) - splambda

def main():
    #x_axis = np.linspace(-2, 2, 1000)
    x_axis = np.linspace(-4, 4, 10000)
    #for i in [2, 0, 3, 4, 5, 6]:
    #    if i !=0:
    #        term_num = i
    #        ratio = 2 / ( term_num - 1 )
    #        y_axis = marginal_prob(x_axis, ratio, term_num) - np.log(i)
    #        plt.plot(x_axis, y_axis, label="DRBM(%d)"%i, linewidth=3.0)
    #    else:
    #        y_axis = marginal_inf(x_axis) - np.log(2)
    #        plt.plot(x_axis, y_axis, label="DRBM(inf)", linewidth=3.0)

    y_axis = marginal_sparse(x_axis, 3, 0)
    plt.plot(x_axis, y_axis, label="SDRBM(3, λ=0)", linewidth=3.0)
    y_axis = marginal_sparse(x_axis, 3, 1)
    plt.plot(x_axis, y_axis, label="SDRBM(3, λ=1)", linewidth=3.0)
    y_axis = marginal_sparse(x_axis, 3, 2)
    plt.plot(x_axis, y_axis, label="SDRBM(3, λ=2)", linewidth=3.0)

    # plt.tick_params(labelbottom=False, bottom=False)
    # plt.tick_params(labelleft=False, left=False)
    # plt.gca().spines["left"].set_visible(False)
    # plt.gca().spines["bottom"].set_visible(False)
    # plt.gca().spines["top"].set_visible(False)
    # plt.gca().spines["right"].set_visible(False)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()