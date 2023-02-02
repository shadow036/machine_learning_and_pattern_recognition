#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 02:43:53 2022

@author: shadow36
"""
from utilities.general_functions import column_vector, row_vector
from utilities.packages_and_constants import np, N_CLASSES, TIED, NAIVE, NAIVE_AND_TIED, NO_REDUCTION, N_FEATURES, LDA, ENTIRE_DATASET_INDEX, DIMENSION_THRESHOLD, FULL
from utilities.dimensionality_reduction import run_PCA, run_LDA
from utilities.model_evaluation import evaluate


def multivariate_gaussian(features, means, covariances, n_features):
    densities = np.zeros(N_CLASSES)
    for i in range(N_CLASSES):
        centered_features = features - means[:, i]
        densities[i] = -0.5*(n_features*np.log(2*np.pi)+(np.linalg.slogdet(covariances[i])[1])+np.dot(np.dot(row_vector(centered_features), np.linalg.inv(covariances[i])), column_vector(centered_features)))
    return densities


def compute_covariance(tr_v, n_features, covariance_type):
    covariances = np.zeros((N_CLASSES, n_features, n_features))
    for i in range(N_CLASSES):
        centered_data = tr_v[i].preprocessed_features - column_vector(tr_v[i].preprocessed_mean)
        covariances[i, :, :] = np.dot(centered_data, centered_data.T)
    if covariance_type == TIED or covariance_type == NAIVE_AND_TIED:
        covariances[0, :, :] += covariances[1, :, :]
        covariances[1, :, :] = covariances[0, :, :]
    if covariance_type == NAIVE or covariance_type == NAIVE_AND_TIED:
        for i in range(N_CLASSES):
            covariances[i, :, :] *= np.eye(n_features)
    for i in range(N_CLASSES):
        if covariance_type == TIED or covariance_type == NAIVE_AND_TIED:
            covariances[i, :, :] /= (tr_v[0].n_samples + tr_v[1].n_samples)
        else:
            covariances[i, :, :] /= tr_v[i].n_samples
    return covariances


def mvg_main(tr_v, te_v, dimensionality_reduction_type, working_point, covariance_type, hy, flag=True):
    print(f'MVG: {dimensionality_reduction_type}, {working_point}, {covariance_type}')
    technique = dimensionality_reduction_type[0]
    threshold = dimensionality_reduction_type[1]
    flag_ = dimensionality_reduction_type[2]
    w = None
    if technique == NO_REDUCTION:
        w = np.eye(hy[N_FEATURES])
    elif technique == LDA:
        w = run_LDA(tr_v, threshold, flag_, hy)
    else:
        w, percentage = run_PCA(tr_v[ENTIRE_DATASET_INDEX], threshold, flag_)
    n_features = w.shape[1]
    for i in range(N_CLASSES):
        tr_v[i].add_features_and_labels(
            np.dot(w.T, tr_v[i].preprocessed_features),
            tr_v[i].labels)
        tr_v[i].add_preprocessed_statistics(n_features=n_features)
    means = np.vstack([tr_v[0].preprocessed_mean, tr_v[1].preprocessed_mean])
    if covariance_type == FULL:
        covariances = [tr_v[0].preprocessed_covariance_matrix, tr_v[1].preprocessed_covariance_matrix]
    else:
        covariances = compute_covariance(tr_v, n_features, covariance_type)
    if flag is False:
        return means, covariances
    te_v.add_features_and_labels(
        np.dot(w.T, te_v.preprocessed_features),
        te_v.labels)
    n_te_samples = te_v.n_samples
    densities = np.zeros((N_CLASSES, n_te_samples))
    for i in range(n_te_samples):
        densities[:, i] = multivariate_gaussian(te_v.preprocessed_features[:, i], means.T, covariances, n_features)
    llr = densities[1, :] - densities[0, :]
    return evaluate(working_point[0], working_point[1], working_point[2], llr, te_v.labels, te_v.n_samples)
    