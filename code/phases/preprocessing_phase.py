#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 19:34:13 2022

@author: shadow36
"""
from utilities.packages_and_constants import ENTIRE_DATASET_INDEX, N_FEATURES, np
from utilities.general_functions import column_vector


def preprocessing_phase(tr, te, hy):
    r_mean = tr[ENTIRE_DATASET_INDEX].mean
    r = tr[ENTIRE_DATASET_INDEX].features - column_vector(r_mean)
    r_min = np.min(r, axis=1)
    r_max = np.max(r, axis=1)
    scaling = [max(r_max[i], abs(r_min[i])) for i in range(hy[N_FEATURES])]
    for t in tr:
        t.add_preprocessed_features(column_vector(scaling), column_vector(r_mean))
        t.add_preprocessed_statistics(hy[N_FEATURES])
    te.add_preprocessed_features(column_vector(scaling), column_vector(r_mean))
