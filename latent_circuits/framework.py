#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 17:23:54 2025

@author: alperencimen
"""

#Import libraries
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tasks import load_task
from model import leaky_current_RNN,leaky_firing_RNN
from utils import training_loop,find_latent_circuit,visualize_latent_circuit
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
#%%
trial = "Trial16"

#%%
"""Importing Task"""
task_path = f"task_data/{trial}/dataset.npz"
dataset = torch.load(task_path)

dataset.visualize_task()
input_dims, output_dims = dataset.get_input_output_dims()
#%%
"""Initialize Model"""
#Model Hyperparameters
hidden_dims = 128
K = 128
device = 'mps'
tau=10

default_seed= 1
torch.manual_seed(default_seed)

model_kwargs = {
    'input_dims': input_dims, 
    'hidden_dims': hidden_dims,
    'output_dims': output_dims, 
    'K': K, 
    'device': device,
    'alpha': dataset.delta_t/tau, 
    'g': 1.5,
    'seed': default_seed
}
model = leaky_firing_RNN(**model_kwargs)
#%%
"""Loadiing Model Parameters"""
model_path = f"model_weights/{trial}/leaky_firing_RNN_{hidden_dims}_{K}.pth"
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint["model_state_dict"])

model.eval()
#%%
inputs = dataset[0]

#%%
Z, kappas, meshgrid = find_latent_circuit(model, dataset,4,[-1,1,200],device)

#%%

visualize_latent_circuit(trial, Z, kappas, meshgrid, [-1,1], 4, "ctx_dep_task", dataset,savefig=False,rotation=[30,30,0])