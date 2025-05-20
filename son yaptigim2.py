#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  8 22:14:04 2025

@author: alperencimen
"""

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
from model import leaky_firing_RNN, leaky_current_RNN
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
# def generate_mante_data_from_conditions(coherences_A, coherences_B, contexts, std=0):
#     # Task constants
#     deltaT = 20.
#     fixation_duration = 100
#     ctx_only_pre_duration = 350
#     stimulus_duration = 800
#     delay_duration = 100
#     decision_duration = 20


#     SCALE = 1e-1
#     SCALE_CTX = 1e-1
#     std_default = 1e-1
#     # decision targets
#     lo = -1
#     hi = 1

#     fixation_duration_discrete = int(fixation_duration // deltaT)
#     ctx_only_pre_duration_discrete = int(ctx_only_pre_duration // deltaT)
#     stimulus_duration_discrete = int(stimulus_duration // deltaT)
#     delay_duration_discrete = int(delay_duration // deltaT)
#     decision_duration_discrete = int(decision_duration // deltaT)
    

#     stim_begin = fixation_duration_discrete + ctx_only_pre_duration_discrete
#     stim_end = stim_begin + stimulus_duration_discrete
#     response_begin = stim_end + delay_duration_discrete
#     total_duration = fixation_duration_discrete + stimulus_duration_discrete + delay_duration_discrete + \
#                         ctx_only_pre_duration_discrete + decision_duration_discrete



#     num_trials = coherences_A.shape[0]
#     inputs_sensory = std * torch.randn((num_trials, total_duration, 2), dtype=torch.float32)
#     inputs_context = torch.zeros((num_trials, total_duration, 2))
#     inputs = torch.cat([inputs_sensory, inputs_context], dim=2)
#     targets = torch.zeros((num_trials, total_duration, 1), dtype=torch.float32)
#     mask = torch.zeros((num_trials, total_duration, 1), dtype=torch.float32)

#     for i in range(num_trials):
#         inputs[i, stim_begin:stim_end, 0] += coherences_A[i] * SCALE
#         inputs[i, stim_begin:stim_end, 1] += coherences_B[i] * SCALE
#         if contexts[i] == 1:
#             inputs[i, fixation_duration_discrete:response_begin, 2] = 1. * SCALE_CTX
#             targets[i, response_begin:] = hi if coherences_A[i] > 0 else lo
#         elif contexts[i] == -1:
#             inputs[i, fixation_duration_discrete:response_begin, 3] = 1. * SCALE_CTX
#             targets[i, response_begin:] = hi if coherences_B[i] > 0 else lo
#         mask[i, response_begin:, 0] = 1
#     return inputs, targets, mask



# #%%
# import itertools

# conditions = itertools.product([-8, 8], [-4, 4], [-1, 1])
# conditions = np.array(list(conditions))
# motion = conditions[:, 0]
# color = conditions[:, 1]
# context = conditions[:, 2]
# choice = np.where(conditions[:, 2] == 1, np.where(conditions[:, 0] > 0, 1, -1), np.where(conditions[:, 1] > 0, 1, -1))
# conditions = np.append(conditions, choice[:, np.newaxis], axis=1)
# print(choice)
# #%%
# x, tt, _ = generate_mante_data_from_conditions(conditions[:, 0], conditions[:, 1], conditions[:, 2])

#%%
#Model Hyperparameters
hidden_dims = 128
K = 4
tau = 10
lr = 1e-3
epoch = 5000

# Don't change unless needed
patience = epoch//10
reg_lambda = 1e-4
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
    'g': None,
    'seed': default_seed
}
"""Do not forget changing the model class here"""
model = leaky_current_RNN(**model_kwargs)

optimizer = optim.Adam(model.parameters(), lr= lr)
criterion = nn.MSELoss()
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                   patience=patience, verbose=False)
batch_size = dataset.data[0].shape[0]
# model, train_losses, acc, lrs, W, P = training_loop(model, dataset, epoch,
#                                           batch_size, model_kwargs['device'], optimizer,
#                                           sched, criterion,lam = reg_lambda,seed=1)

model, train_losses, acc, lrs = training_loop(model, dataset, epoch,
                                          batch_size, model_kwargs['device'], optimizer,
                                          sched, criterion,lam = reg_lambda,seed=1)

#%%
# # Save the visualize_rnn_output plot locally.
# #visualize_rnn_output_path = f'task_train_plots/{trial}/{task_name}_visualize_rnn_output.pdf'
# plt.figure()
# dataset.visualize_rnn_output(model = model,P = torch.Tensor.numpy(P), target = dataset.data[1], loss=train_losses)
# #plt.savefig(visualize_rnn_output_path)
# plt.show()
# plt.close()

# # Save the accuracy and training loss plot locally.
# #accuracy_training_loss_path = f'task_train_plots/{trial}/{task_name}_accuracy_training_loss_plot.pdf'
# plt.figure()
# plt.title("Accuracy and Training Loss")
# plt.plot(acc, label="Accuracy")
# plt.plot(train_losses, label="Training Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Values")
# plt.axhline(y=0, color='r', linestyle='--', label="y=0")
# plt.axhline(y=1, color='g', linestyle='--', label="y=1")
# plt.legend()
# #plt.savefig(accuracy_training_loss_path)
# plt.show()
# plt.close()

#%%

# Save training results (parameters and metrics) to a checkpoint for later use.
# checkpoint_path = f'model_weights/{trial}/{task_name}_BaseRNN_{hidden_dims}_{K}.pth'
# torch.save({
#     'model_state_dict': model.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict(),
#     'scheduler_state_dict': sched.state_dict(),
#     'train_losses': train_losses,
#     'accuracy': acc,
#     'learning_rates': lrs,
#     'epoch': epoch,
#     'model_kwargs': model_kwargs,
#     'P': P
# }, checkpoint_path)
# print(f"Trainin parameters are saved to {checkpoint_path}")


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
#%%
# device = torch.device("cpu")

# inputs_torch = torch.tensor(inputs).to(device)
# outputs_torch = torch.tensor(outputs).to(device)
# # inputs = inputs.numpy()
# # outputs = outputs.numpy()

#%%

o,traj = model.run_rnn(inputs = inputs, device="mps")
#traj = torch.tanh(torch.from_numpy(traj))


#%%

ntime = traj.shape[1]

traj_percond = traj.reshape((decision_array.shape[0], -1))
decision_array = decision_array.reshape((decision_array.shape[0],-1))
linmodel = LinearRegression(fit_intercept=True)
linmodel = linmodel.fit(decision_array,traj_percond)

betas = linmodel.coef_.T.reshape((4, ntime, -1)) #Slopes
intercepts = linmodel.intercept_.T.reshape((ntime, -1)) #intercepts
#%%
betas = np.concatenate([betas, intercepts[np.newaxis, :, :]], axis=0)
print("Betas shape is: ",betas.shape, "The first 4 dimensions are for slopes and the last one is for interception")
#%%
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
plt.savefig("population_dynamics/trajectory_norms.pdf")

#%%

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
# axes[0].scatter(kappas, np.zeros(kappas.shape[0])+np.random.normal(0,0.0001 * kappas.max(),kappas.shape[0]), color="tab:orange",label="Kappas")
# axes[0].plot(centroids, np.zeros_like(centroids), 'rx', markersize=14, markeredgewidth=3, label="End Points",color="purple")
# axes[0].scatter(0,0, color="k",s=200,zorder=10, label="Start Point")

axes[0].set_title("Choice vs Motion")


aaa = plot_projections(axes[1], trials_full_rank, beta_choice, -beta_color, colors, '<--Choice-->', '<--Color-->')
# axes[1].scatter(kappas, np.zeros(kappas.shape[0])+np.random.normal(0,0.0001 * kappas.max(),kappas.shape[0]), color="tab:orange",label="Kappas")
# axes[1].plot(centroids, np.zeros_like(centroids), 'rx', markersize=14, markeredgewidth=3, label="End Points",color="purple")
# axes[1].scatter(0,0, color="k",s=200,zorder=10, label="Start Point")


axes[1].set_title("Choice vs Color")

fig.suptitle("Context = Motion")
fig.text(0.35,0.91,"Dashed lines: Low-Rank RNN. Solid lines: Full-Rank RNN.")
plt.tight_layout() 
plt.savefig("population_dynamics/motion_context.pdf")
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
fig.text(0.35,0.91,"Dashed lines: Low-Rank RNN. Solid lines: Full-Rank RNN.")
plt.tight_layout() 
plt.savefig("population_dynamics/color_context.pdf")
plt.show()

#%%
# Your orthogonal beta vectors (ensure they are column vectors if needed, or handle transposes)
# Assuming beta_choice_ortho etc. are shape (hidden_dims,)
W_rec_effective = model.get_params()["W_rec"]
axes_dict = {
    'choice': beta_choice,
    'motion': beta_motion,
    'color': beta_color,
    'context': beta_context
}
axis_names = list(axes_dict.keys())
num_axes = len(axis_names)
effective_connectivity_matrix = np.zeros((num_axes, num_axes))

for i, name_i in enumerate(axis_names):
    v_i = axes_dict[name_i]
    for j, name_j in enumerate(axis_names):
        v_j = axes_dict[name_j]
        # Effective influence from activity along v_j to future activity along v_i
        # Note: Assumes W_rec operates on tanh(h). If analyzing linearized dynamics around a point,
        # this W_rec might be further multiplied by a diagonal matrix of derivatives of tanh.
        # For a simpler structural view, W_rec itself is often used.
        effective_connectivity_matrix[i, j] = v_i.T @ W_rec_effective @ v_j

plt.figure(figsize=(7,6))
plt.imshow(effective_connectivity_matrix, cmap='coolwarm', origin='upper')
plt.colorbar(label='Effective Connection Strength')
plt.xticks(ticks=np.arange(num_axes), labels=axis_names, rotation=45, ha='right')
plt.yticks(ticks=np.arange(num_axes), labels=axis_names)
plt.title('Effective Connectivity between Task Axes via W_rec')
plt.xlabel('Source Axis (influences...)')
plt.ylabel('Target Axis (...this axis)')
plt.savefig("population_dynamics/axes_effective_connectivity.pdf")
plt.show()

print("Effective Connectivity Matrix (Target_Axis rows, Source_Axis cols):")
print(effective_connectivity_matrix)
#%%

W_rec_effective = model.get_params()['W_rec']
eigenvalues, eigenvectors = np.linalg.eig(W_rec_effective)

# Find the index of the eigenvalue closest to +1.0 (and real)
real_eigenvalues = np.real(eigenvalues)
imag_eigenvalues = np.imag(eigenvalues)

# Filter for real eigenvalues (or those with very small imaginary parts)
# and find the one closest to 1.0
# You might need a tolerance for how close to real it should be
real_indices = np.where(np.abs(imag_eigenvalues) < 1e-6)[0]
if len(real_indices) > 0:
    real_eigenvalues_filtered = real_eigenvalues[real_indices]
    eigenvectors_filtered = eigenvectors[:, real_indices]

    # Find the eigenvalue closest to 1.0 among these
    idx_closest_to_one = np.argmin(np.abs(real_eigenvalues_filtered - 1.0))
    eigenvalue_attractor = real_eigenvalues_filtered[idx_closest_to_one]
    eigenvector_attractor_complex = eigenvectors_filtered[:, idx_closest_to_one]

    # Eigenvectors can be complex, take the real part if the eigenvalue is real
    # (or handle appropriately if it's part of a complex pair very near the real axis)
    eigenvector_attractor = np.real(eigenvector_attractor_complex)
    # Normalize it
    eigenvector_attractor = eigenvector_attractor / np.linalg.norm(eigenvector_attractor)

    print(f"Eigenvalue potentially defining line attractor: {eigenvalue_attractor}")
    # This eigenvector_attractor (shape: hidden_dims,) is the direction of your potential line attractor.
else:
    print("No strictly real eigenvalues found to identify the attractor eigenvector easily.")
    eigenvector_attractor = None # Handle this case

if eigenvector_attractor is not None:
    print(f"Alignment (dot product) of attractor eigenvector with beta_choice_ortho: {np.dot(eigenvector_attractor, beta_choice)}")
    print(f"Alignment (dot product) of attractor eigenvector with beta_motion_ortho: {np.dot(eigenvector_attractor, beta_motion)}")
    print(f"Alignment (dot product) of attractor eigenvector with beta_color_ortho: {np.dot(eigenvector_attractor, beta_color)}")
    print(f"Alignment (dot product) of attractor eigenvector with beta_context_ortho: {np.dot(eigenvector_attractor, beta_context)}")
    
#%%
# ... (previous code: model training, getting betas, getting eigenvector_attractor) ...

# --- Define your filtered trajectory lists for each context ONCE ---
# Make sure 'traj', 'context', and 'choice' are defined in this scope
# 'traj' should be your (num_trials, ntime, hidden_dims) tensor
# 'context' should be your (num_trials,) array of context values (e.g., from decision_array_np[:,2])
# 'choice' should be your (num_trials,) array of choice values (e.g., from decision_array_np[:,3])
# For Motion Context
# (Make sure 'context > 0' correctly identifies motion context based on how you defined 'context')
motion_context_positive_choice_trajs = traj[(context > 0) & (choice > 0)]
motion_context_negative_choice_trajs = traj[(context > 0) & (choice < 0)]
trials_motion_context = [motion_context_positive_choice_trajs,
                         motion_context_negative_choice_trajs]

# For Color Context
# (Make sure 'context < 0' correctly identifies color context)
color_context_positive_choice_trajs = traj[(context < 0) & (choice > 0)]
color_context_negative_choice_trajs = traj[(context < 0) & (choice < 0)]
trials_color_context = [color_context_positive_choice_trajs,
                        color_context_negative_choice_trajs]

# --- Now, the plotting code for activity along the attractor ---
if eigenvector_attractor is not None:
    # Check if eigenvalue_attractor is defined; if not, you might want to define it
    # or ensure it's passed correctly if it comes from a different part of the code.
    # For now, assuming it's defined from your eigenvalue calculation.
    if 'eigenvalue_attractor' not in locals() and 'eigenvalue_attractor' not in globals():
        print("Warning: eigenvalue_attractor not defined, using placeholder for title.")
        eigenvalue_attractor_for_title = "N/A" # Placeholder
    else:
        eigenvalue_attractor_for_title = f"{eigenvalue_attractor:.2f}"


    # activity_along_attractor is not strictly needed here if you project per condition
    # but ntime is useful
    ntime = traj.shape[1]
    time_axis = np.arange(ntime) * dataset.delta_t

    plt.figure(figsize=(10, 6))
    plt.title(f'Activity Projected onto Eigenvector of Eigenvalue ~{eigenvalue_attractor_for_title}')

    # Define colors_plot if not already defined (e.g., from your main trajectory plots)
    # Example:
    if 'colors_plot' not in locals() and 'colors_plot' not in globals():
        cmap = plt.cm.get_cmap('bwr')
        colors_plot = [cmap(0.25), cmap(0.75)] # Blue for cond 0, Red for cond 1 (adjust as needed)


    # Motion Context
    if trials_motion_context[0].shape[0] > 0: # Positive choice trials
        # Ensure eigenvector_attractor is a NumPy array for @ with NumPy array from .numpy()
        eig_vec_np = eigenvector_attractor.numpy() if isinstance(eigenvector_attractor, torch.Tensor) else eigenvector_attractor
        proj_mc_choice_pos = (trials_motion_context[0] @ eig_vec_np).mean(axis=0)
        plt.plot(time_axis, proj_mc_choice_pos, label='Motion Ctx, Choice +', color=colors_plot[0])
    if trials_motion_context[1].shape[0] > 0: # Negative choice trials
        eig_vec_np = eigenvector_attractor.numpy() if isinstance(eigenvector_attractor, torch.Tensor) else eigenvector_attractor
        proj_mc_choice_neg = (trials_motion_context[1] @ eig_vec_np).mean(axis=0)
        plt.plot(time_axis, proj_mc_choice_neg, label='Motion Ctx, Choice -', color=colors_plot[1])

    # Color Context
    if trials_color_context[0].shape[0] > 0: # Positive choice trials
        eig_vec_np = eigenvector_attractor.numpy() if isinstance(eigenvector_attractor, torch.Tensor) else eigenvector_attractor
        proj_cc_choice_pos = (trials_color_context[0] @ eig_vec_np).mean(axis=0)
        plt.plot(time_axis, proj_cc_choice_pos, label='Color Ctx, Choice +', color=colors_plot[0], linestyle='--')
    if trials_color_context[1].shape[0] > 0: # Negative choice trials
        eig_vec_np = eigenvector_attractor.numpy() if isinstance(eigenvector_attractor, torch.Tensor) else eigenvector_attractor
        proj_cc_choice_neg = (trials_color_context[1] @ eig_vec_np).mean(axis=0)
        plt.plot(time_axis, proj_cc_choice_neg, label='Color Ctx, Choice -', color=colors_plot[1], linestyle='--')

    plt.xlabel(f'Time (dt={dataset.delta_t}ms)')
    plt.ylabel('Projection onto Attractor Eigenvector')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig("population_dynamics/activity_along_attractor.pdf")
    plt.show()
    plt.close()
else:
    print("eigenvector_attractor was not found, skipping attractor plot.")