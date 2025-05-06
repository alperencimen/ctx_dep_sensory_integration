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

trial = "Trial5"
#%%

#Task name and version
task_name = 'ctx_dep_mante_task'
version = 'loosen_coherence'


default_seed = np.random.randint(10)      
np.random.seed(default_seed)
     
task_kwargs = {
    'root': '', 
    'scale_context': 1,
    'scale_output': 1,
    'scale_coherences': 1,
    'version': version, 
    'duration': 500, 
    'delta_t': 2,
    'fixation_duration':40,
    'num_trials': 20,
    'input_end_time':400,
    'loosen_after':20,
    'loosen_time':30
}

dataset = load_task(task_name, **task_kwargs)
dataset.visualize_task()

np.save(f'task_data/{trial}/ctx_dep_mante_task_{version}.npy', dataset)

input_dims, output_dims = dataset.get_input_output_dims()
#%%
# version = "vanilla"
# task = np.load(f'task_data/ctx_dep_mante_task_{version}.npy',allow_pickle=True).item()


#%%
#Model Hyperparameters
hidden_dims = 1024
K = 1024
tau = 10
lr = 8e-4
epoch = 3000

# Don't change unless needed
patience = epoch//10
reg_lambda = 1e-3
device = 'mps'

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
    'g': 2,
    'seed': default_seed
}
model = BaseRNN(**model_kwargs)

optimizer = optim.Adam(model.parameters(), lr= lr)
criterion = nn.MSELoss()
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                   patience=patience, verbose=False)
batch_size = dataset.data[0].shape[0]
model, train_losses, acc, lrs, W, P = training_loop(model, dataset, epoch,
                                         batch_size, model_kwargs['device'], optimizer,
                                         sched, criterion,lam = reg_lambda,seed=1)

#%%
# Save the visualize_rnn_output plot locally.
visualize_rnn_output_path = f'task_train_plots/{trial}/{task_name}_visualize_rnn_output.pdf'
plt.figure()
dataset.visualize_rnn_output(model = model,P = P, target = dataset.data[1], loss=train_losses)
plt.savefig(visualize_rnn_output_path)
plt.show()
plt.close()

# Save the accuracy and training loss plot locally.
accuracy_training_loss_path = f'task_train_plots/{trial}/{task_name}_accuracy_training_loss_plot.pdf'
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

#%%

# Save training results (parameters and metrics) to a checkpoint for later use.
checkpoint_path = f'model_weights/{trial}/{task_name}_BaseRNN_{hidden_dims}_{K}.pth'
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': sched.state_dict(),
    'train_losses': train_losses,
    'accuracy': acc,
    'learning_rates': lrs,
    'epoch': epoch,
    'model_kwargs': model_kwargs,
    'P': P
}, checkpoint_path)
print(f"Trainin parameters are saved to {checkpoint_path}")