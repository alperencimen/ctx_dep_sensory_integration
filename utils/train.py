import torch
import time
import wandb
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# make a function to do the training loop, also add a scheduler
# make sure to keep track of both the learning rate and the loss on every epoch
# return the model, the loss, and the learning rate
def training_loop(model, dataset, num_epochs, batch_size, device, optim, sched=None, criterion=nn.MSELoss(),lam = 0,seed = 0, log_wandb: bool = False):
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Train the model, use tqdm to show the progress bar
    pbar = tqdm(range(num_epochs),mininterval=1)
    train_losses, val_losses, lrs = [], [], []
    accs = [];


    for epoch in pbar:

        # Generate a batch of inputs and outputs
        inputs = dataset[0]
        outputs = dataset[1]
        x = torch.from_numpy(inputs).float() # shape (batch_size, seq_len, input_dims)
        y = torch.from_numpy(outputs).float() # shape (batch_size, seq_len, output_dims)

        # move x,y,h to device
        x = x.to(device)
        y = y.to(device)

        # Forward pass
        h = torch.randn(batch_size, model.hidden_dims)
        h = h.to(device)
        output, _ = model(x, h)
        output = torch.stack(output, dim=1)
        
        mask = torch.zeros((1000, 250, 1)).to(device)
        mask[:, 215:, :] = 1
        
        raw_loss = criterion(output, y)
        loss = (raw_loss * mask).sum() / mask.sum()

        if lam == 0:
            loss = loss
        else:
            loss = loss + lam * model.tcr_reg
        
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
        #if (epoch+1) % 100 == 0:
        #    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}', time.strftime("%H:%M:%S", time.localtime()), 'lr', learning_rate)

        # save the loss and the learning rate
        train_losses.append(criterion(output, y).item())
        lrs.append(learning_rate)
        accs.append(acc)
        
        if log_wandb:
            wandb.log({"Epoch": epoch, "Loss": loss.detach().item(), "Accuracy":acc, "Learning Rate":learning_rate,
                       "Regularization":model.tcr_reg.item()})
            
    return model, np.array(train_losses), np.array(accs), np.array(lrs)

def training_loop_weights(model, dataset, num_epochs, batch_size, device, optim, sched=None, criterion=nn.MSELoss(),lam = 0,seed = 0):
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Train the model, use tqdm to show the progress bar
    pbar = tqdm(range(num_epochs),mininterval=1)
    train_losses = []
    accs = [];
    W = np.zeros([num_epochs+1,model.hidden_dims,model.hidden_dims]) + np.nan
    W[0,:,:] = model.get_params()['W_rec']

    for epoch in pbar:

        # Generate a batch of inputs and outputs
        inputs = dataset[0]
        outputs = dataset[1]
        x = torch.from_numpy(inputs).float() # shape (batch_size, seq_len, input_dims)
        y = torch.from_numpy(outputs).float() # shape (batch_size, seq_len, output_dims)

        # move x,y,h to device
        x = x.to(device)
        y = y.to(device)

        # Forward pass
        h = torch.randn(batch_size, model.hidden_dims)
        h = h.to(device)
        output, _ = model(x, h)
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
        
        acc = torch.corrcoef(torch.stack((output.flatten(),y.flatten()),0))[0,1].item()
        pbar.set_postfix({'loss': criterion(output, y).item(), 
                          'acc': acc,
                          'Reg': model.tcr_reg.item()})
        #if (epoch+1) % 100 == 0:
        #    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}', time.strftime("%H:%M:%S", time.localtime()), 'lr', learning_rate)

        # save the loss and the learning rate
        train_losses.append(criterion(output, y).item())
        W[epoch+1,:,:] = model.get_params()['W_rec']
        accs.append(acc)

    return model, np.array(train_losses), np.array(accs), W
    
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


    