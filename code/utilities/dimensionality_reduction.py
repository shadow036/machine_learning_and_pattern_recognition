#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 03:14:39 2022

@author: shadow36
"""
from .general_functions import column_vector, row_vector
from .packages_and_constants import np, sp, N_FEATURES, N_CLASSES, ENTIRE_DATASET_INDEX, PERCENTAGE_THRESHOLD


def compute_between_covariance_matrix(tr_v, hy):
    Sb = np.zeros((hy[N_FEATURES], hy[N_FEATURES]))
    for i in range(N_CLASSES):
        Sb += tr_v[i].n_samples * np.dot(
            column_vector(tr_v[i].preprocessed_mean - tr_v[ENTIRE_DATASET_INDEX].preprocessed_mean), 
            row_vector(tr_v[i].preprocessed_mean - tr_v[ENTIRE_DATASET_INDEX].preprocessed_mean)
            )
    Sb /= tr_v[ENTIRE_DATASET_INDEX].n_samples
    return Sb


def compute_within_covariance_matrix(tr_v, hy):
    Sw = np.zeros((hy[N_FEATURES], hy[N_FEATURES]))
    for i in range(N_CLASSES):
        centered_data = tr_v[i].preprocessed_features - column_vector(tr_v[i].preprocessed_mean)
        Sw += tr_v[i].n_samples * np.dot(centered_data, centered_data.T)
    Sw /= tr_v[ENTIRE_DATASET_INDEX].n_samples
    return Sw


def run_LDA(tr_v, threshold, dimension_or_percentage, hy):
    Sb = compute_between_covariance_matrix(tr_v, hy)
    Sw = compute_within_covariance_matrix(tr_v, hy)
    eigenvalues, eigenvectors = sp.linalg.eigh(Sb, Sw)
    indexes = np.argsort(eigenvalues)[::-1]
    directions = []
    eigenvalues_sum = 0
    if dimension_or_percentage == PERCENTAGE_THRESHOLD:
        for i in indexes:
            eigenvalues_sum += eigenvalues[i]
            if len(directions) == 0: 
                directions = np.array([eigenvectors[:, indexes[i]]])
            else:
                directions = np.vstack([directions, eigenvectors[:, indexes[i]]])
            percentage = 100 * eigenvalues_sum/sum(eigenvalues)
            if percentage >= threshold:
                return directions.T, percentage
    else:
        for i in range(threshold):
            if len(directions) == 0:
                directions = eigenvectors[:, indexes[i]]
            else:
                directions = np.vstack([directions, eigenvectors[:, indexes[i]]])
        return directions.T


def run_PCA(tr_v_dataset, threshold, dimension_or_percentage):
    eigenvalues, eigenvectors = np.linalg.eigh(tr_v_dataset.preprocessed_covariance_matrix)
    indexes = np.argsort(eigenvalues)[::-1]
    principal_components = []
    eigenvalues_sum = 0
    percentage = 0
    if dimension_or_percentage == PERCENTAGE_THRESHOLD:
        for i in indexes:
            eigenvalues_sum += eigenvalues[i]
            if len(principal_components) == 0: 
                principal_components = np.array([eigenvectors[:, indexes[i]]])
            else:
                principal_components = np.vstack([principal_components, eigenvectors[:, indexes[i]]])
            percentage = 100 * eigenvalues_sum/sum(eigenvalues)
            if percentage >= threshold:
                return principal_components.T, percentage
    else:
        for i in range(threshold):
            eigenvalues_sum += eigenvalues[indexes[i]]
            if len(principal_components) == 0:
                principal_components = np.array(eigenvectors[:, indexes[i]])
            else:
                principal_components = np.vstack([principal_components, eigenvectors[:, indexes[i]]])
        percentage = 100 * eigenvalues_sum/sum(eigenvalues)
    return principal_components.T, percentage
    