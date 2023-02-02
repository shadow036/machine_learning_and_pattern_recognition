#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 03:49:37 2022

@author: shadow36
"""

from utilities.packages_and_constants import np, inf, plt


def llr_to_label(threshold, llr):
      return np.array(llr > threshold, dtype=int)


def evaluate(prior, false_negative, false_positive, llr, labels, n_samples):
    threshold = -np.log((prior * false_negative)/((1 - prior) * false_positive))
    predictions = llr_to_label(threshold, llr)
    confusion_matrix = compute_confusion_matrix(predictions, labels, n_samples)
    n_dcf = compute_DCF(prior, false_negative, false_positive, confusion_matrix)
    min_n_dcf = compute_min_DCF(prior, false_negative, false_positive, llr, labels, n_samples)
    return n_dcf, min_n_dcf


def compute_confusion_matrix(predictions, labels, n_samples):
    confusion_matrix = np.zeros((2,2))
    for i in range(n_samples):
        confusion_matrix[predictions[i]][int(labels[i])] += 1    
    return confusion_matrix


def compute_DCF(prior, false_negative, false_positive, confusion_matrix):
    FNR = confusion_matrix[0, 1]/(confusion_matrix[:, 1].sum())
    FPR = confusion_matrix[1, 0]/(confusion_matrix[:, 0].sum())
    u_dcf = (prior * false_negative * FNR) + ((1-prior) * false_positive * FPR)
    n_dcf = u_dcf/min(prior * false_negative, (1 - prior) * false_positive)
    return n_dcf


def compute_min_DCF(prior, false_negative, false_positive, llr, labels, n_samples):
    thresholds = np.linspace(-3, 3, 100)
    min_n_dcf = inf
    for t in thresholds:
        predictions = llr_to_label(t, llr)
        confusion_matrix = compute_confusion_matrix(predictions, labels, n_samples)
        new_n_dcf = compute_DCF(prior, false_negative, false_positive, confusion_matrix)
        if new_n_dcf < min_n_dcf:
            min_n_dcf = new_n_dcf
    return min_n_dcf


def generate_bayes_plot(llr, labels, n_samples, p1, prior, fn, fp):
    thresholds = np.linspace(-3, 3, 100)
    points = np.zeros((3, len(thresholds)))
    index = 0

    for t in thresholds:
        eff_prior = 1/(1 + np.exp(-t))
        predictions = llr_to_label(-t, llr)
        confusion_matrix = compute_confusion_matrix(predictions, labels, n_samples)
        dcf = compute_DCF(prior, fn, fp, confusion_matrix)
        min_dcf = compute_min_DCF(eff_prior, 1, 1, llr, labels, n_samples)
        points[:, index] = [t, dcf, min_dcf]
        index += 1
    plt.figure(str(p1) + ', ')
    plt.plot(points[0, :], points[1, :], label='dcf', color='r')
    plt.plot(points[0, :], points[2, :], label='min dcf', color='b')
    plt.legend()
    plt.xlim(-5, 5)
    plt.ylim(0, 1.5)
    plt.xlabel('negative threshold')
    plt.ylabel('dcf')
    plt.show()
