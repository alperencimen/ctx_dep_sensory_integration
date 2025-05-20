#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 19 19:29:59 2025

@author: alperencimen
"""

from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import maximum_filter
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import math


def find_latent_circuit(model, dataset, K, limits,device):
    results = model.get_params()
    W_rec = results['W_rec']
    W_out = results['W_out']
    
    #Extracting SVD components
    [a,b,c] = np.linalg.svd(W_rec)
    a = a[:,:K]
    b = b[:K]
    c = c[:K,:]
    a =  a*b
    
    #Taking into account for list typed dataset and others.
    if (type(dataset) == list):
        inputs = dataset[0]
        outputs = dataset[1]
    else:
        inputs = dataset.data[0]
        outputs = dataset.data[1]
        
    #Getting neural activities
    o , r =model.run_rnn(inputs = inputs, device = device)
    
    kappas = (c @ r[0,:-1,:].T).T #Calculating kappas
    # reg = LinearRegression().fit(kappas, outputs[0])
    # w_out_kappa = reg.coef_
    # c = w_out_kappa @ c
    # a = a @ np.linalg.inv(w_out_kappa)
    
    # kappas = (c @ r[0,:-1,:].T).T # Initial kappas are normilize in accordance with updated c value.
    
    if (K==1):
        x = np.linspace(limits[0],limits[1],limits[2])
        Z = np.zeros(x.shape[0])
        for i in range(Z.shape[0]):
            Z[i] = -x[i] + c @ np.tanh(a * x[i])
            
        meshgrid = x #In order to return the X. (Note, X is 1 dimensional, not a mesh)
            
    else:    
        def dkappa_func(*args, a, c):
            z = [np.zeros(args[0].shape) for _ in range(K)]
            for i in range(a.shape[0]):
                term = sum(a[i, j] * args[j] for j in range(K))
                for k in range(K):
                    z[k] += c[k, i] * np.tanh(term)
            
            return tuple(z[k] - args[k] for k in range(K))
    
        x = np.array(np.linspace(limits[0], limits[1], limits[2]))
        grids = [x.copy() for _ in range(K)]
        # Creating K-dimensional meshgrid
        meshgrid = np.meshgrid(*grids, indexing='ij')
        
        results = dkappa_func(*meshgrid, a = a, c = c) #Calling dkappa_function
        
        epsilon = 1e-16
        N = np.sqrt(sum(r**2 for r in results)) +epsilon #Epsilon is added in order to prevent division by 0.
        Z = tuple(r / N for r in results) #Normilizing

    return Z, kappas, meshgrid
    
def visualize_latent_circuit(Z, kappas, meshgrid, limits: list, K: int, task_names: str, dataset, rotation: list = [30,30,0], thickness: int = 50, savefig: bool = False, file_name = None, n_components = 3):
    if (K == 1):
        plt.figure()
        plt.title(f"Latent Dynamics | (K = {K}) | Limits: {limits}")
        
        plt.plot(meshgrid, Z, label=f"Latent trajectory")
        plt.scatter(kappas, np.zeros(kappas.shape[0])+np.random.normal(0,0.001 * Z.max(),kappas.shape[0]), color="tab:orange",label="Kappas") #Adding offsets along Y axsis.
        plt.axhline(0 ,color = 'black', ls = '--')
        plt.legend()

    elif (K == 2):
        plt.figure()
        plt.title(f"Latent Dynamics | (K = {K}) | Limits: {limits}")
        
        plt.quiver(meshgrid[0],meshgrid[1],Z[0],Z[1],scale = thickness, label= "Latent trajectories")
        plt.scatter(kappas[:,0],kappas[:,1], 2,c='tab:orange', label="Kappas")
        plt.legend(loc="upper right")

    elif (K == 3): 
        kmeans = KMeans(2**K)
        kmeans.fit(kappas)
        # Get cluster labels and centroids
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_

        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter(kappas[:, 0], kappas[:, 1], kappas[:, 2], c=labels, cmap='rainbow', s = thickness, alpha=0.5)
        ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],  s = thickness*5, alpha=1,color="black",label="Centroids")
       
        ax.set_title(f"Latent Kappas (Clustered) | K = {K}")
        ax.text2D(0.5, 0.95, f"Rotation Parameters: {rotation}", ha='center', fontsize=10, transform=ax.transAxes)
        plt.legend(loc="lower left", bbox_to_anchor=(-0.07,-0.05), framealpha = 0.6)
        ax.view_init(rotation[0],rotation[1],rotation[2])

    else: # Cases where K >= 4.
        if (n_components > K):
            return f"n_components must be less than K. n_components was given as: {n_components}"
        PCA = _PCA(Z=kappas, n_components = n_components)
        Z_matrix = np.vstack(PCA).T
        number_of_components = 2**n_components
        if (number_of_components > dataset.data[1].shape[1]):
            number_of_components = dataset.data[1].shape[1]
            print(f"Number of components is exeeding total input durtion of {dataset.data[1].shape[1]}. So, it is set to {dataset.data[1].shape[1]}")
        kmeans = KMeans(number_of_components) # 2^n_components groups in total
        kmeans.fit(Z_matrix)
       
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
        
        if (n_components == 3):
            ax = plt.figure().add_subplot(projection='3d')
            ax.scatter(Z_matrix[:,0], Z_matrix[:,1], Z_matrix[:,2], c=labels, cmap='rainbow', s = thickness, alpha=0.5)
            ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],  s = thickness*5, alpha=1,color="black",label="Centroids")
           
            ax.set_title(f"Latent Kappas (Clustered) | K = {K}")
            ax.text2D(0.5, 0.95, f"Rotation Parameters: {rotation}", ha='center', fontsize=10, transform=ax.transAxes)
            ax.text2D(0.5, 0.88, f"# of Principle Components: {n_components}", ha='center', fontsize=8, transform=ax.transAxes)
            plt.legend(loc="lower left", bbox_to_anchor=(-0.07,-0.05), framealpha = 0.6)
            ax.view_init(rotation[0],rotation[1],rotation[2])
            
        else:
            num_pairs = n_components // 2
            extra_dim = n_components % 2
            cols = 2
            rows = (num_pairs + extra_dim + 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
            fig.suptitle(f"Principal Component Analysis (PCA) Plots. | K = {K}. | # of Principle Components: {n_components}", fontsize=16)
            axes = axes.flatten()
            
            # Plot each pair of principal components
            for i in range(num_pairs):
                ax = axes[i]
                ax.scatter(PCA[2 * i], PCA[2 * i + 1], cmap='rainbow', alpha=0.6)
                ax.set_xlabel(f'PCA {2 * i}')
                ax.set_ylabel(f'PCA {2 * i + 1}')
                ax.set_title(f'PCA {2 * i} vs PCA {2 * i + 1}')

            # If n_components is odd, plot the last component against PCA0
            if extra_dim:
                ax = axes[num_pairs]
                ax.scatter(PCA[0], PCA[-1], cmap='rainbow', alpha=0.6)
                ax.set_xlabel(f'PCA 0')
                ax.set_ylabel(f'PCA {n_components - 1}')
                ax.set_title(f'PCA 0 vs PCA {n_components - 1}')
            
            # Delete last subplot
            for i in range(num_pairs + extra_dim, len(axes)):
                fig.delaxes(axes[i])
            
    if (savefig):
        task_names = [task_names]
        if (len(task_names) == 1):
            base_path = f'rnnzoo/latent_dynamics/single_tasks/{task_names[0]}/'
        else:
            base_path = f'rnnzoo/latent_dynamics/merged_tasks/{task_names}/'
        if (file_name == None):
            file_name = f"K{K}_network"

        file_path = _get_unique_filename(base_path , file_name)
        plt.savefig(file_path, bbox_inches='tight')
            
    plt.tight_layout()
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

def _PCA(Z, n_components):
    Z = np.nan_to_num(Z, nan=0.0)

    pca = PCA(n_components)
    PC = pca.fit_transform(Z) 
    explained_variance_ratio = pca.explained_variance_ratio_
    print("Explained Variance Ratio of Each Component:", explained_variance_ratio)
    
    return tuple(PC[:, i] for i in range(n_components))

