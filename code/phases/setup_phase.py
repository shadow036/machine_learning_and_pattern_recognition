#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 20:09:17 2022

@author: shadow36
"""
from utilities.packages_and_constants import *


def setup_phase(training_file, test_file):
    hy = [5,  # amount of folds: 80% training - 20% validation
          [(0.9, 1, 10), (0.5, 1, 1), (0.092, 1, 1)],  # format is (prior, false negative cost, false positive cost)
          [NAIVE, TIED, NAIVE_AND_TIED, FULL],
          [1e-6, 1e-5, 1e-4],
          [(0.1, 1), (0.1, 10)],#, (1, 1), (1, 10), (10, 1), (10, 10)],    # svm ck
          [(LDA, 2, DIMENSION_THRESHOLD), (LDA, 3, DIMENSION_THRESHOLD), (LDA, 4, DIMENSION_THRESHOLD),
           (PCA, 95, PERCENTAGE_THRESHOLD), (PCA, 99, PERCENTAGE_THRESHOLD), (PCA, 100, PERCENTAGE_THRESHOLD),
           (NO_REDUCTION, None, None)],
          #[LINEAR_KERNEL, (POLYNOMIAL_KERNEL, 2, 1), (POLYNOMIAL_KERNEL, 3, 1), (GAUSSIAN_KERNEL, 1, None),
           #(GAUSSIAN_KERNEL, 0.2, None)],
          [LINEAR_KERNEL, (POLYNOMIAL_KERNEL, 2, 1), (GAUSSIAN_KERNEL, 1, None)],
          [[CONSTRAINED], [CONSTRAINED, NAIVE], [CONSTRAINED, TIED], [CONSTRAINED, NAIVE, TIED]],
          -1]  # hyperparameters list

    tr = [TrainingSet() for _ in
          range(N_CLASSES + 1)]  # array of training data object (class 0, class 1, entire training dataset)
    te = TestSet()  # test data object
    with open(training_file) as training_file:
        for line in training_file:
            s = [float(f) for f in line.split(',')]
            tr[ENTIRE_DATASET_INDEX].add_sample(s, hy)
            tr[int(s[-1])].add_sample(s, hy)
        for i in range(ENTIRE_DATASET_INDEX, -1, -1):
            tr[i].transpose_and_convert()
            tr[i].add_statistics(hy)
    with open(test_file) as test_file:
        for line in test_file:
            s = [float(f) for f in line.split(',')]
            te.add_sample(s, hy)
    te.transpose_and_convert()

    mvg = np.zeros((len(hy[DIMENSIONALITY_REDUCTION_TYPES]), len(hy[COVARIANCE_TYPES]), len(hy[WORKING_POINTS])),
                  dtype='f, f')  # multivariate gaussian results
    lr = np.zeros((len(hy[LR_LAMBDAS]), len(hy[WORKING_POINTS])), dtype='f, f')  # logistic regression results
    svm = np.zeros((len(hy[SVM_KERNEL_TYPES]), len(hy[SVM_C_K]), len(hy[WORKING_POINTS])),
                  dtype='f, f')  # support vector machines results
    gmm = np.zeros((len(hy[WORKING_POINTS]), int(np.log2(MAX_GMMS)), len(hy[COVARIANCE_TYPES])), dtype='f, f')  # gaussian mixture models results

    re = [mvg, lr, svm, gmm]  # results list
    param = [None, None, None, None]

    return tr, te, hy, re, param
