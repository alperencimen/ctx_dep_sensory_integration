#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 22:10:54 2025

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



#%%

#Task name and version
task_name = 'ctx_dep_mante_task'
version = 'vanilla'


default_seed = np.random.randint(10)      
np.random.seed(default_seed)
     
task_kwargs = {
    'root': '', 
    'version': version, 
    'duration': 500, 
    'delta_t': 2,
    'fixation_duration':200
}

dataset = load_task(task_name, **task_kwargs)
dataset.visualize_task()
input_dims, output_dims = dataset.get_input_output_dims()
#%%
#Model Hyperparameters
hidden_dims = 50
K = 50
tau = 10
lr = 0.0002
epoch = 10000

# Don't change unless needed
patience = epoch//10
reg_lambda = 1e-3
device = 'cpu'

default_seed= 1
torch.manual_seed(default_seed)

#%%
#Model
model_kwargs = {
    'input_dims': input_dims, 
    'hidden_dims': hidden_dims,
    'output_dims': output_dims, 
    'K': K, 
    'device': device,
    'alpha': dataset.delta_t/tau, 
    'g': None,
    'seed': default_seed
}
model = BaseRNN(**model_kwargs)

optimizer = optim.Adam(model.parameters(), lr= lr)
criterion = nn.MSELoss()
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                   patience=patience, verbose=False)
batch_size = dataset.data[0].shape[0]
model, train_losses, acc, lrs = training_loop(model, dataset.data, epoch,
                                         batch_size, model_kwargs['device'], optimizer,
                                         sched, criterion,lam = reg_lambda,seed=1)

#%%
# Save the visualize_rnn_output plot locally.
visualize_rnn_output_path = f'demo/{task_name}_visualize_rnn_output.pdf'
plt.figure()
dataset.visualize_rnn_output(model, train_losses)
plt.savefig(visualize_rnn_output_path)
plt.show()
plt.close()

# Save the accuracy and training loss plot locally.
accuracy_training_loss_path = f'demo/{task_name}_accuracy_training_loss_plot.pdf'
plt.figure()
plt.title("Accuracy and Training Loss")
plt.plot(acc, label="Accuracy")
plt.plot(train_losses, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Values")
plt.axhline(y=0, color='r', linestyle='--', label="y=0")
plt.axhline(y=1, color='g', linestyle='--', label="y=1")
plt.legend()
plt.savefig(accuracy_training_loss_path)
plt.show()
plt.close()

#%%
#Plotting the results without saving on computer.
plt.figure()
plt.title("Accuracy and Training Loss")
plt.plot(acc, label="Accuracy")
plt.plot(train_losses, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Values")
plt.axhline(y=0, color='r', linestyle='--', label="y=0")
plt.axhline(y=1, color='g', linestyle='--', label="y=1")
plt.legend()
plt.savefig(accuracy_training_loss_path)
plt.show()