#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 19 14:48:08 2025

@author: alperencimen
"""

#Import libraries
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tasks import load_task
from model import BaseRNN
from utils import training_loop
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
#%%
#Task name and version
task_name = 'ctx_dep_mante_task'
version = 'vanilla'


default_seed = np.random.randint(107601)  
np.random.seed(default_seed)
     
task_kwargs = {
    'root': '', 
    'seed':default_seed,
    'scale_context': 1,
    'scale_output': 1,
    'scale_coherences': 0.1,
    'version': version, 
    'duration': 500, 
    'delta_t': 2,
    'fixation_duration':60,
    'num_trials': 1000,
    'input_end_time':400,
    'loosen_after':30,
    'loosen_time':20,
    "output_delay":30
}

dataset = load_task(task_name, **task_kwargs)
dataset.visualize_task()

input_dims, output_dims = dataset.get_input_output_dims()
#%%
#Load Model Parameters
