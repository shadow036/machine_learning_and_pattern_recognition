#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 23:33:27 2022

@author: shadow36
"""

from utilities.general_functions import row_vector, column_vector
from utilities.packages_and_constants import np, fmin_l_bfgs_b, GAUSSIAN_KERNEL, POLYNOMIAL_KERNEL
from utilities.model_evaluation import evaluate


def polynomial_kernel(tr_c, te_c, K, d, c):
    return (K**2) + (row_vector(tr_c).dot(column_vector(te_c)) + c) ** d


def gaussian_kernel(tr_c, te_c, K, gamma):
    return (K ** 2) + np.exp(-gamma * (row_vector(tr_c - te_c).dot(column_vector(tr_c - te_c))))


# @ TODO: try to fix this part
def kernel_proxy(ntr, nte, te_v, tr_v, kernel_type, alpha_optimal, z, parameters):
    """
    scores = np.zeros(nte)
    if kernel_type == GAUSSIAN_KERNEL:
        for i in range(nte):
            for j in range(ntr):
                scores[i] += (alpha_optimal[j] * z[j] * gaussian_kernel(tr_v.preprocessed_data[:, j], te_v.preprocessed_data[:, i], parameters[0], parameters[1]))
                print(f'test {i} - training {j}')
    else:
        for i in range(nte):
            for j in range(ntr):
                print(f'{i} - {j}')
                scores[i] += (alpha_optimal[j] * z[j] * polynomial_kernel(tr_v.preprocessed_features[:, j],
                        te_v.preprocessed_features[:, i],
                        parameters[0],
                        parameters[1],
                        parameters[2]))
    return scores"""
    if kernel_type == GAUSSIAN_KERNEL:
        return [
            np.sum(
                [
                    alpha_optimal[j] * z[j] * gaussian_kernel(tr_v.preprocessed_data[:, j], te_v.preprocessed_data[:, i], parameters[0], parameters[1])
                    for j in range(ntr)
                ]
            )
            for i in range(nte)
        ]
    else:
        return [
            np.sum(
                [
                    alpha_optimal[j] * z[j] * polynomial_kernel(tr_v.preprocessed_features[:, j], te_v.preprocessed_features[:, i], parameters[0], parameters[1],
                                                                parameters[2])
                    for j in range(ntr)
                ]
            )
            for i in range(nte)
        ]


def polynomial_gram_matrix(x, K, d, c):
    g = (K ** 2) + (x.T.dot(x) + c) ** d
    return g


def gaussian_gram_matrix(x, K, gamma):
    quadratic_term = (x ** 2).sum(axis=0)
    result = -gamma * ((-2 * np.dot(x.T, x)) + quadratic_term.reshape(-1, 1) + quadratic_term)
    return (K ** 2) + np.exp(result, result)


def objective_function_dual(alpha, H, n):
    ones = column_vector(np.ones(n))
    return 0.5 * row_vector(alpha).dot(H).dot(column_vector(alpha)) - row_vector(alpha).dot(ones), \
        (H.dot(alpha) - column_vector(ones).reshape(n))


def predict_linear_svm(tr_v, te_v, ck, w_point, flag=True):
    print(f'SVM - ck: {ck}, working point: {w_point}')
    ns = tr_v.n_samples
    K = ck[1]
    C = ck[0]
    x_training_extended = np.vstack([tr_v.preprocessed_features, np.full(ns, K)])
    G = np.dot(x_training_extended.T, x_training_extended)
    z = column_vector([2 * tr_v.labels[l] - 1 for l in range(ns)])
    H = list(np.dot(column_vector(z), row_vector(z))) * G
    bounds = [(0, C) for _ in range(ns)]
    alpha_starting = np.ones(ns)
    alpha_optimal, _, _ = fmin_l_bfgs_b(objective_function_dual, alpha_starting,
                                                   args=(H, ns), bounds=bounds, factr=1e12)
    w_optimal_dual = x_training_extended.dot(column_vector(alpha_optimal) * z)
    if flag is False:
        return w_optimal_dual, K
    nse = te_v.n_samples
    x_test_extended = np.vstack([te_v.preprocessed_features, np.full(nse, K)])
    score = row_vector(w_optimal_dual).dot(x_test_extended).reshape(nse)
    return evaluate(w_point[0], w_point[1], w_point[2], score, te_v.labels, nse)


def predict_kernel_svm(tr_v, te_v, CKdc, w_point, flag=False):
    C = CKdc[0]
    K = CKdc[1]
    d = CKdc[2]
    c = CKdc[3]
    print(f'SVM - ck: {(C, K)}, dc: {d, c}, working point: {w_point}')
    if c is None:
        G = gaussian_gram_matrix(tr_v.preprocessed_features, K, d)
    else:
        G = polynomial_gram_matrix(tr_v.preprocessed_features, K, d, c)
    z = column_vector([2 * tr_v.labels[i] - 1 for i in range(tr_v.n_samples)])
    H = list(np.dot(z, row_vector(z))) * G
    bounds = [(0, C) for _ in range(tr_v.n_samples)]
    alpha_starting = np.ones(tr_v.n_samples)
    alpha_optimal, _, _ = fmin_l_bfgs_b(objective_function_dual, alpha_starting,
                                                   args=(H, tr_v.n_samples), bounds=bounds, factr=1e12)
    alpha_optimal = np.array(alpha_optimal)
    if flag is False:
        return (alpha_optimal, z, K, d, c, POLYNOMIAL_KERNEL) if c is not None else (alpha_optimal, z, K, d, GAUSSIAN_KERNEL)
    if c is None:
        scores = kernel_proxy(tr_v.n_samples, te_v.n_samples, te_v, tr_v, GAUSSIAN_KERNEL, alpha_optimal, z, (K, d))
    else:
        scores = kernel_proxy(tr_v.n_samples, te_v.n_samples, te_v, tr_v, POLYNOMIAL_KERNEL, alpha_optimal, z, (K, d, c))
    return evaluate(w_point[0], w_point[1], w_point[2], scores, te_v.labels, te_v.n_samples)

