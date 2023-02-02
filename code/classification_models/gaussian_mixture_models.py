#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 23:42:39 2022

@author: shadow36
"""
from utilities.general_functions import row_vector, column_vector
from utilities.packages_and_constants import *
from utilities.model_evaluation import evaluate


def constant_to_string(mode):
    return f'{"constrained" if CONSTRAINED in mode else ""}{", tied" if TIED in mode else ""}{", naive" if NAIVE in mode else ""}'


def multivariate_gaussian_PDF(f, mean, covariance, ns, nf):
    f_centered = f - column_vector(mean)
    densities = np.zeros(ns)
    for i in range(ns):
        densities[i] = -0.5 * (nf * np.log(2 * np.pi) + (np.linalg.slogdet(covariance)[1]) + np.dot(np.dot(row_vector(f_centered[:, i]), np.linalg.inv(covariance)), column_vector(f_centered[:, i])))
    return densities


def gaussian_mixture_model_PDF(f, gmm, ngmms, ns, hy):
    S = np.zeros((ngmms, ns))
    for j in range(ngmms):
        S[j,:] = multivariate_gaussian_PDF(f, gmm[j][1], gmm[j][2], ns, hy[N_FEATURES]) + np.log(gmm[j][0])
    log_marginal = sp.special.logsumexp(S, axis=0)
    return log_marginal


def constrain_covariance(cov, psi):
    U, s, _ = np.linalg.svd(cov)
    s[s < psi] = psi
    new_cov = np.dot(U, column_vector(s) * U.T)
    return new_cov


def compute_new_parameters(posterior_probability, f, ngmms, nf, ns, psi, mode):
    new_gmms = []
    common_covariance = 0
    if TIED in mode:
        for i in range(ngmms):
            Zg = (posterior_probability[i,:]).sum()
            Fg = np.sum(posterior_probability[i,:] * f, axis=1)
            Sg = np.dot(f, (posterior_probability[i,:] * f).T)
            new_mean = column_vector(Fg/Zg)
            new_covariance = (Sg/Zg) - np.dot(new_mean, row_vector(new_mean))
            common_covariance += (Zg * new_covariance)
    for i in range(ngmms):
        Zg = (posterior_probability[i,:]).sum()
        Fg = np.sum(posterior_probability[i,:] * f, axis=1)
        Sg = np.dot(f, (posterior_probability[i,:] * f).T)
        new_mean = column_vector(Fg/Zg)
        new_covariance = (Sg/Zg) - np.dot(new_mean, row_vector(new_mean))
        new_weight = Zg/ns
        if TIED in mode:
            new_covariance = common_covariance/ns
        if NAIVE in mode:
            new_covariance = new_covariance * np.eye(nf)
        if CONSTRAINED in mode:
            new_covariance = constrain_covariance(new_covariance, psi)
        new_gmms.append((new_weight, new_mean, new_covariance))
    return new_gmms


def EM_algorithm(f, gmm, ngmms, nf, ns, psi, mode):
    old_llr = None
    new_llr = None
    new_gmms = gmm
    S = np.zeros((ngmms, ns))
    while old_llr is None or new_llr - old_llr > 1e-2:
        if new_llr is not None and old_llr is not None:
            print(round(new_llr - old_llr - 1e-5, 4))
        old_llr = new_llr
        # E PHASE
        for j in range(ngmms):
            S[j,:] = multivariate_gaussian_PDF(f, new_gmms[j][1], new_gmms[j][2], ns, nf) + np.log(new_gmms[j][0])
        log_marginals = sp.special.logsumexp(S, axis=0)
        new_llr = log_marginals.sum()/ns
        posterior_probability = np.exp(S - log_marginals)
        # M PHASE
        new_gmms = compute_new_parameters(posterior_probability, f, ngmms, nf, ns, psi, mode)
    print('*')
    return new_gmms, new_llr


def LBG_algorithm(f, gmm, ngmms, nf, ns, psi, alpha, mode):
    GMM_1 = gmm
    GMM_2 = []
    for i in range(ngmms):
        U, s, _ = np.linalg.svd(GMM_1[i][2])
        d = U[:, 0:1] * s[0]**0.5 * alpha
        GMM_2.append((GMM_1[i][0]/2, column_vector(GMM_1[i][1]) + d, GMM_1[i][2]))
        GMM_2.append((GMM_1[i][0]/2, column_vector(GMM_1[i][1]) - d, GMM_1[i][2]))
    new_gmm, llr = EM_algorithm(f, GMM_2, 2 * ngmms, nf, ns, psi, mode)
    return new_gmm, llr


def setup_initial_gmms(means, covariances, nf, psi, modes):
    gmms = [[(1 , means[:, 0], covariances[0, :, :])], [(1, means[:, 1], covariances[1, :, :])]]
    for i in range(2):
        new_cov = constrain_covariance(covariances[i, :, :], psi)
        if NAIVE in modes:
            new_cov = new_cov * np.eye(nf)
        gmms[i][0] = (gmms[i][0][0], gmms[i][0][1], new_cov)
    return gmms


def gmm_main(nste, te_v, gmms, w_point, tr_v, mode, hy):
    print(w_point, mode)
    densities = np.zeros((2, nste))
    for i in range(2):
        densities[i, :] = gaussian_mixture_model_PDF(te_v.preprocessed_features, gmms[i], len(gmms[i]), nste, hy)
    llr = densities[1, :] - densities[0, :]
    next_gmms = deepcopy(gmms)
    for i in range(2):
        next_gmms[i], _ = LBG_algorithm(
            tr_v[i].preprocessed_features, gmms[i], len(gmms[i]), hy[N_FEATURES],
            tr_v[i].n_samples, 0.1, 0.1, mode
        )
    return evaluate(w_point[0], w_point[1], w_point[2], llr, te_v.labels, nste), next_gmms