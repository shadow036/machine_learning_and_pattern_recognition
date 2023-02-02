#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 02:31:16 2022

@author: shadow36
"""

from utilities.packages_and_constants import *


def row_vector(data):
    return np.array(data).reshape((1, len(data)))


def column_vector(data):
    return np.array(data).reshape((len(data), 1))


def show_dataset_info(hy, tr, flag):
    fig_dist, axs_dist = plt.subplots(2, 4)
    fig_dist.suptitle('feature distributions - ' + ('original' if flag == ORIGINAL else 'preprocessed') + ' data',
                      fontsize=20)
    fig_cov, axs_cov = plt.subplots(1, 3)
    fig_cov.suptitle('covariance matrices - ' + ('original' if flag == ORIGINAL else 'preprocessed') + ' data',
                     fontsize=20)
    for i in range(N_CLASSES + 1):
        im = axs_cov[i % (N_CLASSES + 1)].imshow(tr[i].covariance_matrix[:, :]) if flag == ORIGINAL \
            else axs_cov[i % (N_CLASSES + 1)].imshow(tr[i].preprocessed_covariance_matrix[:, :])
        axs_cov[i % (N_CLASSES + 1)].set_xticks(np.arange(hy[N_FEATURES]), labels=range(hy[N_FEATURES]))
        axs_cov[i % (N_CLASSES + 1)].set_yticks(np.arange(hy[N_FEATURES]), labels=(FEATURES_NAMES_SHORTENED if i == 0
                                                                                   else range(hy[N_FEATURES])))
        for j in range(hy[N_FEATURES]):
            range_ = [min(tr[ENTIRE_DATASET_INDEX].features[j, :]),
                      max(tr[ENTIRE_DATASET_INDEX].features[j, :])] \
                if flag == ORIGINAL else \
                [min(tr[ENTIRE_DATASET_INDEX].preprocessed_features[j, :]),
                 max(tr[ENTIRE_DATASET_INDEX].preprocessed_features[j, :])]
            axs_dist[j // 4, j % 4].hist(tr[i].features[j, :] if flag == ORIGINAL
                                         else tr[i].preprocessed_features[j, :],
                                         bins=50,
                                         linewidth=0.5 + int(i == ENTIRE_DATASET_INDEX),
                                         alpha=0.5,
                                         range=range_,
                                         edgecolor='black',
                                         label=LABELS[i],
                                         color=COLORS[i],
                                         density=True)
            axs_dist[j // 4, j % 4].legend()
            axs_dist[j // 4, j % 4].set_title(FEATURES_NAMES[j])
            for k in range(hy[N_FEATURES]):
                text = axs_cov[i % (N_CLASSES + 1)].text(j, k, round(
                    tr[i].covariance_matrix[j, k]) if flag == ORIGINAL else float(
                    round(tr[i].preprocessed_covariance_matrix[j, k], 3)),
                                                         ha="center", va="center", color="w")
        axs_cov[i % (N_CLASSES + 1)].set_title(LABELS[i])
    plt.show()
    if flag == ORIGINAL:
        fig, axs = plt.subplots(1, 3)
        fig.suptitle('correlation matrices', fontsize=20)
        for i in range(N_CLASSES + 1):
            im = axs[i % (N_CLASSES + 1)].imshow(
                tr[i].correlation_matrix[:, :] if flag == ORIGINAL else tr[i].preprocessed_correlation_matrix[:, :])
            axs[i % (N_CLASSES + 1)].set_xticks(np.arange(hy[N_FEATURES]), labels=range(hy[N_FEATURES]))
            axs[i % (N_CLASSES + 1)].set_yticks(np.arange(hy[N_FEATURES]),
                                                labels=(FEATURES_NAMES_SHORTENED if i == 0 else range(hy[N_FEATURES])))
            for j in range(hy[N_FEATURES]):
                for k in range(hy[N_FEATURES]):
                    text = axs[i % (N_CLASSES + 1)].text(j, k, float(
                        round(tr[i].correlation_matrix[j, k], 3)) if flag == ORIGINAL else float(
                        round(tr[i].preprocessed_correlation_matrix[j, k], 3)),
                                                         ha="center", va="center", color="w")
            axs[i % (N_CLASSES + 1)].set_title(LABELS[i])
        plt.show()


def generate_K_folds(tr_dataset, folds, f, hy):
    delimiter = tr_dataset.n_samples // folds
    # features to be included in the temporary training set
    reference_features = tr_dataset.preprocessed_features[:, :(delimiter * f)]
    # labels to be included in the temporary training set
    reference_labels = tr_dataset.labels[:(delimiter * f)]
    if f < folds - 1:
        reference_features = np.hstack([
            reference_features,
            tr_dataset.preprocessed_features[:, (delimiter * (f + 1)):]
        ])
        reference_labels = np.concatenate((reference_labels, tr_dataset.labels[(delimiter * (f + 1)):]))
    tr_v = [TrainingSet(), TrainingSet(), TrainingSet()]
    for i in range(N_CLASSES):
        tr_v[i].add_features_and_labels(reference_features[:, reference_labels == i],
                                        reference_labels[reference_labels == i])
        tr_v[i].add_preprocessed_statistics(hy[N_FEATURES])
    tr_v[ENTIRE_DATASET_INDEX].add_features_and_labels(reference_features, reference_labels)
    tr_v[ENTIRE_DATASET_INDEX].add_preprocessed_statistics(hy[N_FEATURES])
    te_v = TestSet()
    te_v.add_features_and_labels(  # generate validation setvd
        tr_dataset.preprocessed_features[:, (delimiter * f):(delimiter * (f + 1))],
        tr_dataset.labels[(delimiter * f):(delimiter * (f + 1))]
    )
    return tr_v, te_v
