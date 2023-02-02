#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 19:52:56 2022

@author: shadow36
"""
from classification_models.gaussian_classifiers import multivariate_gaussian
from classification_models.support_vector_machines import predict_linear_svm
from classification_models.gaussian_mixture_models import gmm_main

from utilities.packages_and_constants import *
from utilities.model_evaluation import evaluate, generate_bayes_plot
from utilities.general_functions import row_vector


def testing_phase(te, hy, parameters, params, tr):

    mvg_p = parameters[0]
    lr_p = parameters[1]
    svm_p = parameters[2]
    gmms = parameters[3]

    densities = np.zeros((N_CLASSES, te.n_samples))
    for i in range(te.n_samples):
        densities[:, i] = multivariate_gaussian(te.preprocessed_features[:, i], mvg_p[0].T, mvg_p[1], hy[N_FEATURES])
    llr = densities[1, :] - densities[0, :]
    print(f'mvg: {evaluate(params[MVG][0][2][0], params[MVG][0][2][1], params[MVG][0][2][2], llr, te.labels, te.n_samples)}')
    generate_bayes_plot(llr, te.labels, te.n_samples, 'mvg', params[MVG][0][2][0], params[MVG][0][2][1], params[MVG][0][2][2])

    score = (np.dot(lr_p[0], te.preprocessed_features) + lr_p[1]).flatten()
    print(f'logistic regression: {evaluate(params[LR][0][1][0], params[LR][0][1][1], params[LR][0][1][2], score, te.labels, te.n_samples)}')
    generate_bayes_plot(score, te.labels, te.n_samples, 'lr', params[LR][0][1][0], params[LR][0][1][1], params[LR][0][1][2])

    test_extended = np.vstack([te.preprocessed_features, np.full(te.n_samples, svm_p[1])])
    score = row_vector(svm_p[0]).dot(test_extended).reshape(te.n_samples)
    print(f'svm: {evaluate(params[SVM][0][1][0], params[SVM][0][1][1], params[SVM][0][1][2], score, te.labels, te.n_samples)}')

    results = None
    for am in range(int(np.log2(MAX_GMMS))):
        for g in range(len(gmms)):
            results, gmms[g] = gmm_main(te.n_samples, te, gmms[g], params[GMM][0][2], tr, params[GMM][0][1], hy)
    print(f'gmm: {results}')

    return

