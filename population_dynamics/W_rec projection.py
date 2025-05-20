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
from model import leaky_current_RNN,leaky_firing_RNN
from utils import training_loop,visualize_neuron_activities,visualize_sorted_activities
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
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
outputs = dataset[1]
ctx_decisions = dataset[2]
decision_array = dataset[4]
decision_array = np.array(decision_array)


motion = decision_array[:,0]
color = decision_array[:,1]
context = decision_array[:,2]
choice = decision_array[:,3]

o,traj = model.run_rnn(inputs = inputs, device = device)
#%%

ntime = traj.shape[1]

traj_percond = traj.reshape((decision_array.shape[0], -1))
decision_array = decision_array.reshape((decision_array.shape[0],-1))
linmodel = LinearRegression(fit_intercept=True)
linmodel = linmodel.fit(decision_array,traj_percond)

betas = linmodel.coef_.T.reshape((4, ntime, -1)) #Slopes
intercepts = linmodel.intercept_.T.reshape((ntime, -1)) #intercepts

betas = np.concatenate([betas, intercepts[np.newaxis, :, :]], axis=0)
print("Betas shape is: ",betas.shape, "The first 4 dimensions are for slopes and the last one is for interception")

tmaxes = []
labels = ['motion', 'color', 'context', 'choice', 'intercept']
for i in range(5):
    norms = np.linalg.norm(betas[i], axis=1)
    p,=plt.plot(norms, label=labels[i], lw=3)
    tmaxes.append(np.argmax(norms))
    plt.plot([tmaxes[-1]], [norms[tmaxes[-1]]], marker='*', c=p.get_color(), markersize=10)
plt.legend(bbox_to_anchor=(1, .8))
plt.ylabel('norm')
plt.xlabel('time (dt=5ms)')


beta_motion = betas[0, tmaxes[0]]
beta_color = betas[1, tmaxes[1]]
beta_context = betas[2, tmaxes[2]]
beta_choice = betas[3, tmaxes[3]]
Bmat = np.vstack([beta_choice, beta_motion, beta_color, beta_context]).T
print(Bmat.shape)
BmatQ, _ = np.linalg.qr(Bmat)
beta_choice = BmatQ[:, 0]
beta_motion = BmatQ[:, 1]
beta_color = BmatQ[:, 2]
beta_context = BmatQ[:, 3]

#%%
W_rec = model.get_params()["W_rec"]
alpha_val = model.alpha 
W_eff = (1 - alpha_val) * np.eye(hidden_dims) + alpha_val * W_rec

W_rec_in_beta_basis = BmatQ.T @ W_eff @ BmatQ

plt.figure(figsize=(6,5))
plt.imshow(W_rec_in_beta_basis, cmap='coolwarm', aspect='auto')
plt.colorbar(label="Effective Weight")
coding_labels = ['choice', 'motion', 'color', 'context'] # Match BmatQ column order
plt.xticks(ticks=np.arange(len(coding_labels)), labels=coding_labels, rotation=45)
plt.yticks(ticks=np.arange(len(coding_labels)), labels=coding_labels)
plt.xlabel("Input Coding Direction")
plt.ylabel("Output Coding Direction")
plt.title("W_rec in the basis of orthogonalized coding directions")
plt.savefig(f"model_lct_and_dynamics/{trial}/W_rec_in_beta_basis.pdf")
plt.show()

print("W_rec_in_beta_basis:\n", W_rec_in_beta_basis)

#%%

