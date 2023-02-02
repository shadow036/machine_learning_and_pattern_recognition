import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.optimize import fmin_l_bfgs_b
from math import inf
import scipy as sp

# general constants
N_CLASSES = 2
ENTIRE_DATASET_INDEX = 2
N_CLASSIFICATION_MODELS = 4

# constants used to access the "result" list defined in "setup_phase.py"
MVG = 0
LR = 1
SVM = 2
GMM = 3

# constants used to access the "hyperparameters" list define in "setup_phase.py"
FOLDS = 0
WORKING_POINTS = 1
COVARIANCE_TYPES = 2
LR_LAMBDAS = 3
SVM_C_K = 4
DIMENSIONALITY_REDUCTION_TYPES = 5
SVM_KERNEL_TYPES = 6
COVARIANCE_TYPES_GMMS = 7

N_FEATURES = 8

# covariance types (mvg)
NAIVE = 0
TIED = 1
NAIVE_AND_TIED = 2
FULL = 3

# covariance types (gmm)
CONSTRAINED = 4


MAX_GMMS = 8

# dimensionality reduction types
PCA = 0
LDA = 1
NO_REDUCTION = 2

# dimensionality reduction threshold metrics
DIMENSION_THRESHOLD = 0
PERCENTAGE_THRESHOLD = 1

# kernel types
LINEAR_KERNEL = 0
POLYNOMIAL_KERNEL = 1
GAUSSIAN_KERNEL = 2

# relationship feature index - feature name
FEATURES_NAMES = ['mean integrated profile',
                  'standard deviation integrated profile',
                  'excess kurtosis integrated profile',
                  'skewness integrated profile',
                  'mean DM-SNR',
                  'standard deviation DM-SNR',
                  'excess kurtosis DM-SNR',
                  'skewness DM-SNR']

FEATURES_NAMES_SHORTENED = ['mean IR',
                            'standard deviation IR',
                            'excess kurtosis IR',
                            'skewness IR',
                            'mean DM-SNR',
                            'standard deviation DM-SNR',
                            'excess kurtosis DM-SNR',
                            'skewness DM-SNR']

# rescaling constants
RESCALING_MIN = -1
RESCALING_MAX = 1

# distinction between original and preprocessed data
ORIGINAL = 0
PREPROCESSED = 1

LABELS = ['class 0', 'class 1', 'dataset']
COLORS = ['red', 'green', 'white']


class Hyperparameters:
    def __init__(self, folds, working_points, covariance_types, lr_lambda, svm_c_k, dimensionality_reduction,
                 svm_kernel_type, max_gmms):
        self.folds = folds
        self.working_points = working_points,
        self.covariance_types = covariance_types
        self.lr_lambda: lr_lambda
        self.svm_c_k = svm_c_k
        self.dimensionality_reduction = dimensionality_reduction
        self.svm_kernel_type = svm_kernel_type
        self.max_gmms = max_gmms


class TestSet:
    def __init__(self):
        self.preprocessed_features = None
        self.n_samples = 0
        self.features = []
        self.labels = []

    def add_sample(self, sample, hy):
        self.features.append(sample[:-1])
        if hy[N_FEATURES] == -1:
            hy[N_FEATURES] = len(sample) - 1
        self.labels.append(sample[-1])
        self.n_samples += 1

    def add_features_and_labels(self, features, labels):
        self.preprocessed_features = features
        self.labels = labels
        self.n_samples = len(labels)

    def transpose_and_convert(self):
        self.features = np.array(self.features).T
        self.labels = np.array(self.labels)

    def add_preprocessed_features(self, scaling, mean):
        # centering
        self.preprocessed_features = self.features - mean
        # rescaling in the range [-1, 1] with centered mean
        self.preprocessed_features = self.preprocessed_features / scaling
        # changing the dataset entries from [-1, 1] to [-10, 10]
        self.preprocessed_features *= 10
    # WRONG APPROACH
    # self.preprocessed_features =
    # RESCALING_MIN + (RESCALING_MAX - RESCALING_MIN) * (self.features - column_vector(min_)) /
    # np.tile(column_vector(max_ - min_), (1, self.n_samples)) # simple rescaling in the range [-1, 1]


class TrainingSet(TestSet):
    def __init__(self):
        super().__init__()
        self.preprocessed_mean = None
        self.correlation_matrix = None
        self.mean = None
        self.covariance_matrix = None
        self.preprocessed_covariance_matrix = None

    def add_statistics(self, hy):
        self.mean = np.mean(self.features, axis=1)
        # do not use "column_vector" function due to the creation of a cyclic dependency;
        # the averaging is useless for computing the covariance of the whole dataset since it's 0 by definition
        # but it can be useful for the datasets labeled 0 and 1
        self.covariance_matrix = np.dot(self.features - np.reshape(self.mean, (hy[N_FEATURES], 1)),
                                        self.features.T - self.mean) / self.n_samples
        diagonal_array = np.diag(self.covariance_matrix) ** 0.5
        diagonal_matrix = np.linalg.inv(np.diag(diagonal_array))
        self.correlation_matrix = np.dot(np.dot(diagonal_matrix, self.covariance_matrix), diagonal_matrix)

    def add_preprocessed_statistics(self, n_features):
        self.preprocessed_mean = self.preprocessed_features.mean(axis=1)
        self.preprocessed_covariance_matrix = np.dot(
            self.preprocessed_features - np.reshape(self.preprocessed_features.mean(axis=1), (n_features, 1)),
            self.preprocessed_features.T - self.preprocessed_mean) / self.n_samples
        # the correlation matrix is the same of the initial dataset (without preprocessing)
