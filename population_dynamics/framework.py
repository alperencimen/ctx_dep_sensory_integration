#Importing required libraries.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from model.base_rnn import BaseRNN
import torch
from tasks.ctx_dep_mante_task.data import CtxDepManteTask
from sklearn.cluster import KMeans
#%%
""" Task initialization part. Don't run this line unless needed as saved data could be used."""

# version = "vanilla"
# task_kwargs = {
#     'root': '', 
#     'version': version,
#     'duration': 500, 
#     'delta_t': 2,
#     'fixation_duration':50,
#     'scale_context': 1,
#     'scale_coherences':8,
#     'scale_output':1
# } 
# task = CtxDepManteTask(**task_kwargs) 
# task.visualize_task()

#%%
"""Importing mante task."""

version = "vanilla"
task = np.load(f'task_data/ctx_dep_mante_task_{version}.npy',allow_pickle=True).item()

#%%
inputs = task.data[0]
outputs = task.data[1]
ctx_decisions = task.data[2]
decision_array = task.data[5]
decision_array = np.array(decision_array)

motion = decision_array[:,0]
color = decision_array[:,1]
context = decision_array[:,2]
choice = decision_array[:,3]
#%%
"""Initialization of the model"""
default_seed= 1
torch.manual_seed(default_seed)
input_dims, output_dims = inputs.shape[2], outputs.shape[2]

model_kwargs = {
    'input_dims': input_dims, 
    'hidden_dims': 1024,
    'output_dims': output_dims, 
    'K': 4, 
    'device': 'mps',
    'alpha': task.delta_t/10, 
    'g': 2,
    'seed': default_seed
}

model = BaseRNN(**model_kwargs)
#%%
"""Importing model weights"""
checkpoint = torch.load("model_weights/ctx_dep_mante_task_BaseRNN_1024_4.pth")

# Load the stored model parameters.
model.load_state_dict(checkpoint['model_state_dict'])

P = checkpoint['P']

# # Retrieve additional training artifacts if needed.
# train_losses = checkpoint['train_losses']
# accuracy = checkpoint['accuracy']
# learning_rates = checkpoint['learning_rates']
# start_epoch = checkpoint['epoch']


#%%
"""Forward propagating the initialized model"""

o,traj = model.run_rnn(inputs,device="mps",P=P, target = outputs)
#%%
#traj = torch.tanh(torch.from_numpy(traj)).detach().numpy()

#%%

W_rec = model.get_params()["W_rec"]
#%%
def stretch_values(x, old_min=-0.1, old_max=0.1, new_min=-2.68, new_max=2.68):
    return (x - old_min) * (new_max - new_min) / (old_max - old_min) + new_min

#%%
def get_latent(network,outp):
    #W_rec_full_rank = network.wrec.detach().numpy()
    W_rec_full_rank = network

    [a_full,b_full,c_full] = np.linalg.svd(W_rec_full_rank)
    a_full = a_full[:,:1]
    b_full = b_full[:1]
    c_full = c_full[:1,:]
    c_full =  np.diag(b_full) @ c_full

    r_full = traj[0]

    kappas = (c_full @ r_full[:-1,:].T).T #Calculating kappas


    reg = LinearRegression().fit(kappas, outp[0])
    w_out_kappa = reg.coef_
    c_full = w_out_kappa @ c_full

    a_full = a_full @ np.linalg.inv(w_out_kappa)

    kappas = (c_full @ r_full[:-1,:].T).T # Initial kappas are normilize in accordance with updated c value.
    new_kappas = stretch_values(kappas, kappas.min(), kappas.max())
    
    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(new_kappas)
    centroids = kmeans.labels_
    
    new_centroids = stretch_values(centroids, centroids.min(), centroids.max())
      
    return new_kappas, new_centroids

#%%

kappas, centroids = get_latent(W_rec,outputs)

#%%
ntime = traj.shape[1]

traj_percond = traj.reshape((decision_array.shape[0], -1))
decision_array = decision_array.reshape((decision_array.shape[0],-1))
linmodel = LinearRegression(fit_intercept=True)
linmodel = linmodel.fit(decision_array, traj_percond)

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
    plt.plot(norms, label=labels[i], lw=3)
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
        ax.plot((tr @ axis1).mean(axis=0), (tr @ axis2).mean(axis=0), c=colors[i], lw=4)
        if (np.max((tr @ axis1).mean(axis=0)) >= np.max((tr @ axis2).mean(axis=0))):
            ll.append(np.max((tr @ axis1).mean(axis=0)))
        else:
            ll.append(np.max((tr @ axis2).mean(axis=0)))

        if trials_other is not None:
            ax.plot((trials_other[i] @ axis1).mean(axis=0), (trials_other[i] @ axis2).mean(axis=0), c=colors[i], ls=':', lw=4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(lab1)
    ax.set_ylabel(lab2)
#%%
trials_full_rank = [traj[(context == 1) & (choice > 0)],
          traj[(context == 1) & (choice < 0)]]

cmap = matplotlib.cm.get_cmap('bwr')
colors = [cmap(0), cmap(255)]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))  
cmap = matplotlib.cm.get_cmap('bwr')
colors = [cmap(0), cmap(255)]


aa = plot_projections(axes[0], trials_full_rank, beta_choice, -beta_motion, colors, '<--Choice-->', '<--Motion-->')
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


cmap = matplotlib.cm.get_cmap('bwr')
colors = [cmap(0), cmap(255)]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))  
cmap = matplotlib.cm.get_cmap('bwr')
colors = [cmap(0), cmap(255)]


plot_projections(axes[0], trials_full_rank, beta_choice, -beta_color, colors, '<--Choice-->', '<--Color-->')
# axes[0].scatter(kappas, np.zeros(kappas.shape[0])+np.random.normal(0,0.0001 * kappas.max(),kappas.shape[0]), color="tab:orange",label="Kappas")
# axes[0].plot(centroids, np.zeros_like(centroids), 'rx', markersize=14, markeredgewidth=3, label="End Points",color="purple")
# axes[0].scatter(0,0, color="k",s=200,zorder=10, label="Start Point")

axes[0].set_title("Choice vs Motion")


plot_projections(axes[1], trials_full_rank, beta_choice, -beta_motion, colors, '<--Choice-->', '<--Motion-->')
# axes[1].scatter(kappas, np.zeros(kappas.shape[0])+np.random.normal(0,0.0001 * kappas.max(),kappas.shape[0]), color="tab:orange",label="Kappas")
# axes[1].plot(centroids, np.zeros_like(centroids), 'rx', markersize=14, markeredgewidth=3, label="End Points",color="purple")
# axes[1].scatter(0,0, color="k",s=200,zorder=10, label="Start Point")

axes[1].set_title("Choice vs Color")


fig.suptitle("Context = Color")
fig.text(0.35,0.91,"Dashed lines: Low-Rank RNN. Solid lines: Full-Rank RNN.")
plt.tight_layout() 
plt.savefig("population_dynamics/color_context.pdf")
plt.show()

