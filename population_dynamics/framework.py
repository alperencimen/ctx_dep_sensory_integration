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
from utils import training_loop
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
#%%
trial = "Trial1"

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
K = 4
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
model = leaky_current_RNN(**model_kwargs)
#%%
"""Loadiing Model Parameters"""
model_path = f"model_weights/{trial}/leaky_current_RNN_{hidden_dims}_{K}.pth"
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
#traj = torch.tanh(torch.from_numpy(traj))


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
#plt.savefig(f"model_dynamics/{trial}/trajectory_norms.pdf")



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


def plot_projections(ax, trials, axis1, axis2, colors, lab1, lab2, trials_other=None):
    """Modified function to take an axis argument (ax) for subplot plotting."""
    for i, tr in enumerate(trials):
        print(i,tr.shape)
        num_averaged_trials = tr.shape[0]
        print(f"Plotting condition {i} (Color: {colors[i]}): Averaging {num_averaged_trials} trials. Projection: {lab1} vs {lab2}")
        ax.plot((tr @ axis1).mean(axis=0), (tr @ axis2).mean(axis=0), c=colors[i], lw=4)
       

        if trials_other is not None:
            ax.plot((trials_other[i] @ axis1).mean(axis=0), (trials_other[i] @ axis2).mean(axis=0), c=colors[i], ls=':', lw=4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(lab1)
    ax.set_ylabel(lab2)
#%%
trials_full_rank = [traj[(context == 1) & (choice > 0)],
          traj[(context == 1) & (choice < 0)]]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))  
cmap = matplotlib.cm.get_cmap('bwr')
colors = [cmap(0), cmap(255)]


plot_projections(axes[0], trials_full_rank, beta_choice, -beta_motion, colors, '<--Choice-->', '<--Motion-->')
axes[0].scatter(kappas, np.zeros(kappas.shape[0])+np.random.normal(0,0.0001 * kappas.max(),kappas.shape[0]), color="tab:orange",label="Kappas")
axes[0].plot(centroids, np.zeros_like(centroids), 'rx', markersize=14, markeredgewidth=3, label="End Points",color="purple")
# axes[0].scatter(0,0, color="k",s=200,zorder=10, label="Start Point")

axes[0].set_title("Choice vs Motion")


plot_projections(axes[1], trials_full_rank, beta_choice, -beta_color, colors, '<--Choice-->', '<--Color-->')
# axes[1].scatter(kappas, np.zeros(kappas.shape[0])+np.random.normal(0,0.0001 * kappas.max(),kappas.shape[0]), color="tab:orange",label="Kappas")
axes[1].plot(centroids, np.zeros_like(centroids), 'rx', markersize=14, markeredgewidth=3, label="End Points",color="purple")
# axes[1].scatter(0,0, color="k",s=200,zorder=10, label="Start Point")

axes[1].set_title("Choice vs Color")


fig.suptitle("Context = Motion")
#fig.text(0.35,0.91,"Dashed lines: Low-Rank RNN. Solid lines: Full-Rank RNN.")
plt.tight_layout() 
plt.savefig(f"model_dynamics/{trial}/motion_context.pdf")
plt.show()

#%%
trials_full_rank = [traj[(context == -1) & (choice > 0)],
          traj[(context == -1) & (choice < 0)]]


fig, axes = plt.subplots(1, 2, figsize=(12, 5))  
cmap = matplotlib.cm.get_cmap('bwr')
colors = [cmap(0), cmap(255)]


plot_projections(axes[0], trials_full_rank, beta_choice, -beta_color, colors, '<--Choice-->', '<--Color-->')
# axes[0].scatter(kappas, np.zeros(kappas.shape[0])+np.random.normal(0,0.0001 * kappas.max(),kappas.shape[0]), color="tab:orange",label="Kappas")
# axes[0].plot(centroids, np.zeros_like(centroids), 'rx', markersize=14, markeredgewidth=3, label="End Points",color="purple")
# axes[0].scatter(0,0, color="k",s=200,zorder=10, label="Start Point")

axes[0].set_title("Choice vs Color")


plot_projections(axes[1], trials_full_rank, beta_choice, -beta_motion, colors, '<--Choice-->', '<--Motion-->')
# axes[1].scatter(kappas, np.zeros(kappas.shape[0])+np.random.normal(0,0.0001 * kappas.max(),kappas.shape[0]), color="tab:orange",label="Kappas")
# axes[1].plot(centroids, np.zeros_like(centroids), 'rx', markersize=14, markeredgewidth=3, label="End Points",color="purple")
# axes[1].scatter(0,0, color="k",s=200,zorder=10, label="Start Point")

axes[1].set_title("Choice vs Motion")


fig.suptitle("Context = Color")
#fig.text(0.35,0.91,"Dashed lines: Low-Rank RNN. Solid lines: Full-Rank RNN.")
plt.tight_layout() 
plt.savefig(f"model_dynamics/{trial}/color_context.pdf")
plt.show()
