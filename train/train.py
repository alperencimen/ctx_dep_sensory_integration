#Import libraries
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tasks import load_task
from model import leaky_current_RNN, leaky_firing_RNN
from utils import training_loop
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
#%%
#Task name and version
task_name = 'ctx_dep_mante_task'
version = 'vanilla'


default_seed = np.random.randint(50612937)  
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
    "output_delay":30
}

dataset = load_task(task_name, **task_kwargs)
dataset.visualize_task()

input_dims, output_dims = dataset.get_input_output_dims()
#%%
trial = "Trial16"

#%%
#Saving Task Parameters
task_path = f"task_data/{trial}/dataset.npz"
torch.save(dataset,task_path)


#%%
#Model Hyperparameters
hidden_dims = 128
K = 128
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
    'g': 1.5,
    'seed': default_seed
}
model = leaky_firing_RNN(**model_kwargs)

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
# Save the visualize_rnn_output plot
visualize_rnn_output_path = f'task_train_plots/{trial}/{task_name}_visualize_rnn_output.pdf'

plt.figure()
dataset.visualize_rnn_output(model = model, loss=train_losses)
plt.savefig(visualize_rnn_output_path)
plt.show()
plt.close()

# Save the accuracy and training loss plot 
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
#Save training results (parameters and metrics) to a checkpoint for later use.
checkpoint_path = f'model_weights/{trial}/leaky_firing_RNN_{hidden_dims}_{K}.pth'
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': sched.state_dict(),
    'train_losses': train_losses,
    'accuracy': acc,
    'learning_rates': lrs,
    'epoch': epoch,
    'model_kwargs': model_kwargs
}, checkpoint_path)
print(f"Trainin parameters are saved to {checkpoint_path}")

