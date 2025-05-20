#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 19 19:31:12 2025

@author: alperencimen
"""

import matplotlib.pyplot as  plt
import numpy as np
import os

def find_neuron_activities(model, dataset,device):
    #Taking into account for list typed dataset and others.
    if (type(dataset) == list):
        inputs = dataset[0]
        outputs = dataset[1]
    else:
        inputs = dataset.data[0]
        outputs = dataset.data[1]
    
    #Obtaining neural activities
    o, r = model.run_rnn(inputs = inputs, device = device)
    activities = r[0].T
    
    return activities

def visualize_neuron_activities(activities, dataset, task_names: list, savefig: bool = False, file_name = None):
    #Taking into account for list typed dataset and others.
    if (type(dataset) == list):
        inputs = dataset[0]
        outputs = dataset[1]
    else:
        inputs = dataset.data[0]
        outputs = dataset.data[1]
        
    fig, axs = plt.subplots(2,1,figsize=(10,8),sharex=False)
    
    axs[0].plot(inputs[0]) #Visualization for the first trail
    axs[0].set_title("Task's Input")
    axs[0].set_ylabel("Output")
    axs[0].set_xlabel("Time (ms)")
    axs[0].grid(True)
    
    cax = axs[1].imshow(activities, aspect='auto', cmap='RdBu_r',
                    interpolation='none', vmin=-1, vmax=1)

    axs[1].set_title("Neural Firings")
    axs[1].set_xlabel("Time (ms)")
    axs[1].set_ylabel("Recurrent units")

    # Adjust colorbar
    cbar = plt.colorbar(cax)
    cbar.set_label("Activation Intensity")
    cbar.set_ticks([-1, 0, 1])
    plt.tight_layout()
    
    if (savefig):
        task_names = [task_names]
        if (len(task_names) == 1):
            base_path = f'rnnzoo/neuron_activities/single_tasks/{task_names[0]}/'
        else:
            base_path = f'rnnzoo/neuron_activities/merged_tasks/{task_names}/'
        if (file_name == None):
            file_name = f"Network_activities"

        file_path = _get_unique_filename(base_path , file_name)
        plt.savefig(file_path, bbox_inches='tight')
        
    plt.show()

def visualize_sorted_activities(activities, dataset, sorting_interval: list, task_names: list, savefig: bool = False, file_name = None):
    #Taking into account for list typed dataset and others.
    if (type(dataset) == list):
        inputs = dataset[0]
        outputs = dataset[1]
    else:
        inputs = dataset.data[0]
        outputs = dataset.data[1]
        
    if (len(sorting_interval) != 2):
        return f"Sorting interval must contain 2 values seperated with a comma. However, passed: {sorting_interval}"
       
    if (sorting_interval[0] >= sorting_interval[1]):
        return f"Sorting Interval must be entered in increasing order. For instance: [50,80]. However, passed {sorting_interval}"
    
    
    fig, axs = plt.subplots(2,1,figsize=(10,8),sharex=False)
    
    axs[0].plot(inputs[0]) #Visualization for the first trial
    axs[0].set_title(f"Task's Input | Sorting Interval: {sorting_interval}")
    axs[0].set_ylabel("Output")
    axs[0].set_xlabel("Time (ms)")
    axs[0].grid(True)
    
    #Sorting focused interval's neural activities in decreasing order.
    sorted_activities = activities[:, sorting_interval[0]:sorting_interval[1]]
    activity_levels = np.sum(sorted_activities, axis=1)
    sorted_indices = np.argsort(activity_levels)[::-1] 
    sorted_recurrent_activities = activities[sorted_indices]
    
    axs[0].vlines(x=sorting_interval[0], ymin=np.min(inputs[0]), ymax=np.max(inputs[0]), 
           linestyles="solid", colors="purple", linewidth=4,label=f"x={sorting_interval[0]} Start point of the interval.")
           
    axs[0].vlines(x=sorting_interval[1], ymin=np.min(inputs[0]), ymax=np.max(inputs[0]), 
           linestyles="solid", colors="tab:orange", linewidth=4,label=f"x={sorting_interval[1]} End point of the interval.")
    
    axs[0].set_xticks(list(axs[0].get_xticks()[1:-1]) + [sorting_interval[0],sorting_interval[1]])

    
    cax = axs[1].imshow(sorted_recurrent_activities, aspect='auto', cmap='RdBu_r',
                    interpolation='none', vmin=-1, vmax=1)


    axs[1].set_title(f"Neural Firings | {sorting_interval}(ms) Interval is sorted in decreasing activity order.")
    axs[1].set_xlabel("Time (ms)")
    axs[1].set_ylabel("Recurrent units")
    
    axs[1].vlines(x=sorting_interval[0], ymin= 0, ymax= sorted_recurrent_activities.shape[0]-1, 
           linestyles="solid", colors="purple", linewidth=4,label=f"x={sorting_interval[0]} Start point of the interval.")
           
    axs[1].vlines(x=sorting_interval[1], ymin= 0, ymax= sorted_recurrent_activities.shape[0]-1, 
           linestyles="solid", colors="tab:orange", linewidth=4,label=f"x={sorting_interval[1]} End point of the interval.")
    
    axs[1].set_xticks(list(axs[0].get_xticks()[1:-1]) + [sorting_interval[0],sorting_interval[1]])

    # Adjust colorbar
    cbar = plt.colorbar(cax)
    cbar.set_label("Activation Intensity")
    cbar.set_ticks([-1, 0, 1])
    plt.tight_layout()
    
    axs[0].legend(loc = "upper right",bbox_to_anchor = (1.3,1),framealpha=1)
    axs[1].legend(loc="upper right",bbox_to_anchor=(1.6,1))

    if (savefig):
        task_names = [task_names]
        if (len(task_names) == 1):
            base_path = f'rnnzoo/neuron_activities/single_tasks/{task_names[0]}/'
        else:
            base_path = f'rnnzoo/neuron_activities/merged_tasks/{task_names}/'
        if (file_name == None):
            file_name = f"Sorted_network_activities"

        file_path = _get_unique_filename(base_path , file_name)
        plt.savefig(file_path, bbox_inches='tight')
    
    plt.show()
    
# Helper function to generate a unique file name
def _get_unique_filename(base_path, file_name, extension=".pdf"):
    counter = 1
    file_path = f"{base_path}{file_name}{extension}"
    
    # Check if file already exists, increment counter if needed
    while os.path.exists(file_path):
        file_path = f"{base_path}{file_name}_{counter}{extension}"
        counter += 1
    
    return file_path
