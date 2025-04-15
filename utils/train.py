import torch
import time
#import wandb
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# make a function to do the training loop, also add a scheduler
# make sure to keep track of both the learning rate and the loss on every epoch
# return the model, the loss, and the learning rate
def training_loop(model, dataset, num_epochs, batch_size, device, optim, sched=None,
                  criterion=nn.MSELoss(),lam = 0,seed = 0, training_weights = False,
                  log_wandb: bool = False, visualize_steps: bool = False, visualize_step_size: int = 1000):
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Train the model, use tqdm to show the progress bar
    pbar = tqdm(range(num_epochs),mininterval=1)
    train_losses, val_losses, lrs = [], [], []
    accs = [];
    
    #Taking into account for list typed dataset and others.
    if (type(dataset) == list):
        inputs = dataset[0]
        outputs = dataset[1]
    else:
        inputs = dataset.data[0]
        outputs = dataset.data[1]
    
    
    if (training_weights):
        W_in = np.zeros([num_epochs+1,inputs.shape[2],model.hidden_dims]) + np.nan
        W_rec = np.zeros([num_epochs+1,model.hidden_dims,model.hidden_dims]) + np.nan
        W_out = np.zeros([num_epochs+1,model.hidden_dims,outputs.shape[2]]) + np.nan
       
        W_in[0,:,:] = model.get_params()['W_in'] #Initial network weights.
        W_rec[0,:,:] = model.get_params()['W_rec'] 
        W_out[0,:,:] = model.get_params()['W_out']
    else:
        W_in = np.empty([num_epochs+1,inputs.shape[2],model.hidden_dims]) 
        W_rec = np.empty([num_epochs+1,model.hidden_dims,model.hidden_dims]) 
        W_out = np.empty([num_epochs+1,model.hidden_dims,outputs.shape[2]]) 
        
    P = np.eye(model.hidden_dims).astype(np.float32)
    P = torch.from_numpy(P).to(model.device)

    P = P.detach()
    P.requires_grad_(False)
    P = P / 3000
    
    target = dataset.data[1]
    target = torch.from_numpy(target)
    target = target.to(torch.float32).to(model.device)


    for epoch in pbar:
        x = torch.from_numpy(inputs).float() # shape (batch_size, seq_len, input_dims)
        y = torch.from_numpy(outputs).float() # shape (batch_size, seq_len, output_dims)

        # move x,y,h to device
        x = x.to(device)
        y = y.to(device)

        # Forward pass
        h = torch.randn(batch_size, model.hidden_dims)
        h = h.to(device)
        
        
        
        output, _,P = model(x, h, P, target)
        
        
        output = torch.stack(output, dim=1)
        if lam == 0:
            loss = criterion(output, y)
        else:
            loss = criterion(output, y) + lam * model.tcr_reg
        
        # Backward and optimize
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optim.step()
        sched.step(loss)
        
        # Corr coef : is this fast?
        acc = torch.corrcoef(torch.stack((output.flatten(),y.flatten()),0))[0,1].item()
        
        learning_rate = optim.param_groups[0]['lr']
        pbar.set_postfix({'loss': criterion(output, y).item(), 
                          'acc': acc,
                          'lr': learning_rate, 
                          'Reg': model.tcr_reg.item()})
       
        # save the loss and the learning rate
        train_losses.append(criterion(output, y).item())
        lrs.append(learning_rate)
        accs.append(acc)
        
       
        if log_wandb: #Logging parameters to WandB when True.
            wandb.log({"Epoch": epoch, "Loss": loss.detach().item(), "Accuracy":acc, "Learning Rate":learning_rate,
                       "Regularization":model.tcr_reg.item()})
      
        
        if (training_weights): #Saving network's weights at every each pass when true.
            W_in[epoch+1,:,:] = model.get_params()['W_in'] #Initial network weights.
            W_rec[epoch+1,:,:] = model.get_params()['W_rec'] 
            W_out[epoch+1,:,:] = model.get_params()['W_out'] 
        
        #Plotting results while training continues
        if (visualize_steps and type(dataset) != list):
            if (epoch % visualize_step_size == 0 or epoch == num_epochs-1):
                _visualize_rnn_output_single_task(model, dataset, train_losses, epoch, accs)
                plt.show()
                plt.close()

        elif (visualize_steps and type(dataset) == list):
            if (epoch % visualize_step_size == 0 or epoch == num_epochs-1):
                _visualize_rnn_output_for_combined_tasks(model, dataset, epoch, accs, train_losses)
                plt.tight_layout()
                plt.show()
                plt.close()
                  
    return model, np.array(train_losses), np.array(accs), np.array(lrs), (np.array(W_in), np.array(W_rec), np.array(W_out)), P

def compute_mse(model, task, n_tasks = 1, batch_size = 1, device = "cpu", visualize = False):
    # n_tasks denotes number of tasks to solve then take the average of their MSEs
    mse = []
    
    # For loop may be slowing down?
    for i in range(n_tasks):
        inputs, expected_outputs = task.gen_batch(batch_size=batch_size)  # Visualize one trial for simplicity
        # Below is likely the costly part.
        predicted_output, _ = model.run_rnn(inputs, device)
            
        ss_res = ((expected_outputs - predicted_output) ** 2).sum()
        mse.append(ss_res/expected_outputs.shape[1])
        
    if visualize:
        outputs_expanded = expected_outputs[np.newaxis, :]  # Add batch dimension to outputs
        predicted_output_expanded = predicted_output[np.newaxis, :]  # Add batch dimension to predicted_output
    
        task.visualize_rnn_output(predicted_output_expanded[0],outputs_expanded[0])
        plt.show()
        
    return sum(mse)/len(mse)

def _visualize_rnn_output_single_task(model, dataset, train_losses, epoch, acc):
    #Plotting loss and accuracy
    plt.figure()
    plt.title(f"Accuracy and Training Loss | Epoch: {epoch}")
    plt.plot(acc, label="Accuracy")
    plt.plot(train_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Values")
    plt.axhline(y=0, color='r', linestyle='--', label="y=0")
    plt.axhline(y=1, color='g', linestyle='--', label="y=1")
    plt.legend()
    plt.show()
    #Calling task's visualize_rnn_output code with current epoch.
    dataset.visualize_rnn_output(model, train_losses, current_epoch = epoch)

def _visualize_rnn_output_for_combined_tasks(model, dataset, epoch, acc, train_losses):
    #plot accuracy and training loss plot.
    plt.figure()
    plt.title(f"Accuracy and Training Loss | Epoch: {epoch}")
    plt.plot(acc, label="Accuracy")
    plt.plot(train_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Values")
    plt.axhline(y=0, color='r', linestyle='--', label="y=0")
    plt.axhline(y=1, color='g', linestyle='--', label="y=1")
    plt.legend()
    plt.show()

    #Plotting RNN output.
    dataset_inputs = dataset[0]
    dataset_outputs = dataset[1]
    
    predicted_outputs, _ = model.run_rnn(dataset_inputs, device=model.device)

    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        predicted_outputs, _ = model.run_rnn(dataset_inputs, device=model.device)


    num_trials_to_plot = 1 # Visualize the output with desired number of trials.
    
    fig, axs = plt.subplots(num_trials_to_plot, 1, figsize=(35, num_trials_to_plot*15))
    fig.suptitle(f'Model Output vs. Actual Output | Epoch: {epoch}',fontsize=25)
    
    if num_trials_to_plot == 1:
        axs = [axs]

    for i in range(0,num_trials_to_plot):
        # If there's only one trial, axs is not a list, so we don't use axs[i]
        axs[i].plot(predicted_outputs[i], label='Model Output', linestyle="-")
        axs[i].plot(dataset_outputs[i], label='Task Output', linestyle="--")
        axs[i].set_xlabel('Time Steps')
        axs[i].set_ylabel('Output')
        axs[i].legend(fontsize=20)