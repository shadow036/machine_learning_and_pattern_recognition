#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 04:02:39 2022

@author: shadow36
"""
from utilities.packages_and_constants import np, fmin_l_bfgs_b, ENTIRE_DATASET_INDEX
from utilities.model_evaluation import evaluate


def objective_function(v, tr_data, zis, ns, n_features, my_lambda):
    w = v[:-1]
    b = v[-1]
    # gradient_w = np.zeros(n_features)
    # gradient_b = 0
    exp = (-zis * (np.dot(w, tr_data) + b))
    partial_logistic_loss = np.logaddexp(0, exp).sum() + ns - 1
    total_logistic_loss = (((my_lambda/2) * (w**2).sum()) + partial_logistic_loss/ns)
    """
    for i in range(ns):
        # attempt to avoid overflow:
        # if exponent is very large, neglect the 1 at the denominator
        # and simplify the exponential both at the numerator and at the denominator
        if exp[i] >= 4:
            gradient_w -= (zis[i] * tr_data[:, i])/ns
            gradient_b -= (zis[i]/ns)
        else:
            # also divide by ns at each cycle in order to keep the numbers small
            gradient_w -= ((zis[i] * tr_data[:, i] * np.exp(exp[i]))/((1 + np.exp(exp[i])) * ns))
            gradient_b -= (zis[i] * np.exp(exp[i])/((1 + np.exp(exp[i])) * ns))
    gradient_w = (my_lambda * w) - gradient_w
    return total_logistic_loss, np.hstack([gradient_w, gradient_b])
    """
    return total_logistic_loss


def predict_logistic_regression(tr_v, te_v, n_features, w_point, my_lambda, flag=True):
    print(f'LG: {w_point}, {my_lambda}')
    tr_data = tr_v.preprocessed_features
    tr_labels = tr_v.labels
    ns = tr_v.n_samples
    zis = np.array([(2 * tr_labels[i] - 1) for i in range(ns)])
    result, minim, _ = fmin_l_bfgs_b(objective_function, np.zeros(n_features+1),
                                     args=(tr_data, zis, ns, n_features, my_lambda), approx_grad=True)
    # print(minim)
    w = result[:-1]
    b = result[-1]
    if flag is False:
        return w, b
    te_data = te_v.preprocessed_features
    te_labels = te_v.labels
    score = (np.dot(w, te_data) + b).flatten()
    return evaluate(w_point[0], w_point[1], w_point[2], score, te_labels, te_v.n_samples)
