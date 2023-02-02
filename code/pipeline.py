#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 01:58:01 2022

@author: shadow36
"""

from utilities.packages_and_constants import ORIGINAL, PREPROCESSED, plt, deepcopy
from utilities.general_functions import show_dataset_info
from phases.training_phase import training_phase
from phases.preprocessing_phase import preprocessing_phase
from phases.testing_phase import testing_phase
from phases.setup_phase import setup_phase

plt.close('all')

tr, te, hy, re, param = setup_phase('./../data/train_pulsar', './../data/test_pulsar')
# show_dataset_info(hy, tr, ORIGINAL)
preprocessing_phase(tr, te, hy)
# show_dataset_info(hy, tr, PREPROCESSED)

testing_parameters = training_phase(tr, hy, re, param)
testing_phase(te, hy, testing_parameters, param, deepcopy(tr))
