import matplotlib.pyplot as plt
import numpy as np
from utilities.general_functions import generate_K_folds, column_vector
from utilities.packages_and_constants import *
from utilities.model_evaluation import evaluate, generate_bayes_plot
from utilities.dimensionality_reduction import run_LDA

from classification_models.gaussian_classifiers import mvg_main
from classification_models.gaussian_mixture_models import gmm_main, LBG_algorithm, setup_initial_gmms, \
    constant_to_string
from classification_models.logistic_regression import predict_logistic_regression
from classification_models.support_vector_machines import predict_linear_svm, predict_kernel_svm


def training_phase(tr, hy, re, param):
    folds = hy[FOLDS]
    validation_data = [generate_K_folds(tr[ENTIRE_DATASET_INDEX], folds, f, hy) for f in range(folds)]

    # GAUSSIAN CLASSIFIER ----------------------------------------------------------------------------------------------
    results = [
        mvg_main(deepcopy(v[0]), deepcopy(v[1]), dim_red, w_point, cov_type, hy)
        for dim_red in hy[DIMENSIONALITY_REDUCTION_TYPES]
        for cov_type in hy[COVARIANCE_TYPES]
        for w_point in hy[WORKING_POINTS]
        for v in validation_data
    ]
    j = 0
    min_ = inf
    for i_dim_red in range(len(hy[DIMENSIONALITY_REDUCTION_TYPES])):
        for i_cov_type in range(len(hy[COVARIANCE_TYPES])):
            for i_w_point in range(len(hy[WORKING_POINTS])):
                for f in range(folds):
                    re[MVG][i_dim_red, i_cov_type, i_w_point] = \
                        (re[MVG][i_dim_red, i_cov_type, i_w_point][0] + results[j][0],
                         re[MVG][i_dim_red, i_cov_type, i_w_point][1] + results[j][1])
                    j = j + 1
                re[MVG][i_dim_red, i_cov_type, i_w_point] = (re[MVG][i_dim_red, i_cov_type, i_w_point][0] / folds,
                                                             re[MVG][i_dim_red, i_cov_type, i_w_point][1] / folds)
                if re[MVG][i_dim_red, i_cov_type, i_w_point][1] <= min_:
                    min_ = re[MVG][i_dim_red, i_cov_type, i_w_point][1]
                    param[MVG] = ((hy[DIMENSIONALITY_REDUCTION_TYPES][i_dim_red],
                                   hy[COVARIANCE_TYPES][i_cov_type],
                                   hy[WORKING_POINTS][i_w_point]), min_)


    tr_copy = deepcopy(tr)
    w = run_LDA(tr_copy, 3, DIMENSION_THRESHOLD, hy)
    new_tr_copy = w.T.dot(tr_copy[ENTIRE_DATASET_INDEX].preprocessed_features)
    ax = plt.figure().add_subplot(projection='3d')
    ax.scatter(new_tr_copy[0, :], new_tr_copy[1, :], new_tr_copy[2, :],
                c=['red' if l == 0 else 'green' for l in tr_copy[ENTIRE_DATASET_INDEX].labels], alpha=0.1,
               label=['class 0' if l == 0 else 'class 1' for l in tr_copy[ENTIRE_DATASET_INDEX].labels])
    plt.show()
    
    # LOGISTIC REGRESSION ----------------------------------------------------------------------------------------------
    results = [
        predict_logistic_regression(v[0][ENTIRE_DATASET_INDEX], v[1], hy[N_FEATURES], w_point, lam)
        for lam in hy[LR_LAMBDAS]
        for w_point in hy[WORKING_POINTS]
        for v in validation_data
    ]
    j = 0
    min_ = inf
    for i_lam in range(len(hy[LR_LAMBDAS])):
        for i_w_point in range(len(hy[WORKING_POINTS])):
            for iv in range(folds):
                re[LR][i_lam, i_w_point] = (re[LR][i_lam, i_w_point][0] + results[j][0],
                                            re[LR][i_lam, i_w_point][1] + results[j][1])
                j = j + 1
            re[LR][i_lam, i_w_point] = (re[LR][i_lam, i_w_point][0] / folds, re[LR][i_lam, i_w_point][1] / folds)
            if re[LR][i_lam, i_w_point][1] <= min_:
                min_ = re[LR][i_lam, i_w_point][1]
                param[LR] = ((hy[LR_LAMBDAS][i_lam], hy[WORKING_POINTS][i_w_point]), min_)
    # generate_bayes_plot(llr, info_te.get('labels'), info_te.get('cardinality'), lr_type, lam, w_point)

    # SVMs ----------------------------------------------------------------------------------
    results = [
        predict_linear_svm(v[0][ENTIRE_DATASET_INDEX], v[1], ck, w_point)
        for ck in hy[SVM_C_K]
        for w_point in hy[WORKING_POINTS]
        for v in validation_data
    ]
    j = 0
    min_ = inf
    for i_ck in range(len(hy[SVM_C_K])):
        for i_w_point in range(len(hy[WORKING_POINTS])):
            for iv in range(folds):
                re[SVM][0, i_ck, i_w_point] = (re[SVM][0, i_ck, i_w_point][0] + results[j][0],
                                               re[SVM][0, i_ck, i_w_point][1] + results[j][1])
                j = j + 1
            re[SVM][0, i_ck, i_w_point] = (re[SVM][0, i_ck, i_w_point][0] / folds,
                                           re[SVM][0, i_ck, i_w_point][1] / folds)
            if re[SVM][0, i_ck, i_w_point][1] <= min_:
                min_ = re[SVM][0, i_ck, i_w_point][1]
                param[SVM] = ((hy[SVM_C_K][i_ck], hy[WORKING_POINTS][i_w_point], 'linear'), min_)

    min_ = inf
    results = [
        predict_kernel_svm(v[0][ENTIRE_DATASET_INDEX], v[1], [ck[0], ck[1], dc[0], dc[1]], w_point)
        for dc in hy[SVM_KERNEL_TYPES][1:]
        for ck in hy[SVM_C_K]
        for w_point in hy[WORKING_POINTS]
        for v in validation_data
    ]
    for i_dc in range(1, len(hy[SVM_KERNEL_TYPES])):
        for i_ck in range(len(hy[SVM_C_K])):
            for i_w_point in range(len(hy[WORKING_POINTS])):
                for iv in range(folds):
                    re[SVM][i_dc, i_ck, i_w_point] = (re[SVM][i_dc, i_ck, i_w_point][0] + results[j][0],
                                                      re[SVM][i_dc, i_ck, i_w_point][1] + results[j][1])
                    j = j + 1
                re[SVM][i_dc, i_ck, i_w_point] = (re[SVM][i_dc, i_ck, i_w_point][0] / folds,
                                                  re[SVM][i_dc, i_ck, i_w_point][1] / folds)
                if re[SVM][i_dc, i_ck, i_w_point][1] <= min_:
                    min_ = re[SVM][i_dc, i_ck, i_w_point][1]
                    param[SVM] = (
                        (
                            (
                                hy[SVM_C_K][i_ck], hy[SVM_KERNEL_TYPES][i_dc[1]]
                            )
                            if hy[SVM_KERNEL_TYPES][i_dc] == GAUSSIAN_KERNEL
                            else
                            (
                                hy[SVM_C_K][i_ck],
                                (
                                    hy[SVM_KERNEL_TYPES][i_dc][1], hy[SVM_KERNEL_TYPES][i_dc][2]
                                )
                            ),
                            hy[WORKING_POINTS][i_w_point],
                            'gaussian'
                            if hy[SVM_KERNEL_TYPES][i_dc] == GAUSSIAN_KERNEL
                            else
                            'polynomial'),
                        min_
                    )

    # GMM ------------------------------------------------------------------------------------------
    gmms = [
        setup_initial_gmms(
            np.hstack(
                (column_vector(v[0][0].preprocessed_mean), column_vector(v[0][1].preprocessed_mean))
            ), np.array([v[0][0].preprocessed_covariance_matrix, v[0][1].preprocessed_covariance_matrix]), hy[N_FEATURES],
            0.1, m
        )
        for m in hy[COVARIANCE_TYPES_GMMS] for v in validation_data
    ]

    min_ = inf
    next_g = None
    for am in range(int(np.log2(MAX_GMMS))):
        for g in range(len(gmms)):
            for i_w_point in range(len(hy[WORKING_POINTS])):
                results, next_g = gmm_main(
                    validation_data[g % folds][1].n_samples,
                    validation_data[g % folds][1],
                    gmms[g], hy[WORKING_POINTS][i_w_point],
                    validation_data[g % folds][0],
                    hy[COVARIANCE_TYPES_GMMS][g // folds],
                    hy
                )
                re[GMM][i_w_point, am, g // folds] = (re[GMM][i_w_point, am, g // folds][0] + results[0],
                                                      re[GMM][i_w_point, am, g // folds][1] + results[1])
                if g % folds == folds - 1:
                    re[GMM][i_w_point, am, g // folds] = (re[GMM][i_w_point, am, g // folds][0] / folds,
                                                          re[GMM][i_w_point, am, g // folds][1] / folds)
                    if re[GMM][i_w_point, am, g // folds][1] < min_:
                        min_ = re[GMM][i_w_point, am, g // folds][1]
                        param[GMM] = ((2 ** (am+1), hy[COVARIANCE_TYPES_GMMS][g // folds], hy[WORKING_POINTS][i_w_point]), min_)
            gmms[g] = next_g
        print(f'done with am = {am**2}')

    # training everything on the entire training set

    means, covariances = mvg_main(deepcopy(tr), None, param[MVG][0][0], param[MVG][0][1], param[MVG][0][2], hy, False)
    w, b = predict_logistic_regression(tr[ENTIRE_DATASET_INDEX], None, hy[N_FEATURES], param[LR][0][1],
                                       param[LR][0][0], flag=False)
    w_optimal_dual, K = predict_linear_svm(tr[ENTIRE_DATASET_INDEX], None, param[SVM][0][0], param[SVM][0][1], False)

    gmms = setup_initial_gmms(np.hstack((column_vector(tr[0].preprocessed_mean), column_vector(tr[1].preprocessed_mean))),
                              np.array([tr[0].preprocessed_covariance_matrix, tr[1].preprocessed_covariance_matrix]), hy[N_FEATURES], 0.1, param[GMM][0][1])

    return (means, covariances), (w, b), (w_optimal_dual, K), gmms
