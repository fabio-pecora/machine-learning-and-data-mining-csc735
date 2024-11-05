#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


# In[3]:


# Load the dataset
file_path = 'C:\\Users\\lynet\\Downloads\\vehicle.csv' 
vehicle_data = pd.read_csv(file_path)

# Display the first few rows
print(vehicle_data.head())


# In[5]:


# Drop non-numeric columns or handle them as needed
vehicle_data = vehicle_data.select_dtypes(include=[np.number]).dropna()

# Normalize the data 
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(vehicle_data)


# In[7]:


def spectral_clustering_experiment(data, gamma_values, n_clusters):
    silhouette_scores = []
    
    for gamma in gamma_values:
        clustering = SpectralClustering(n_clusters=n_clusters, affinity='rbf', gamma=gamma)
        labels = clustering.fit_predict(data)
        
        # Calculate the silhouette score
        score = silhouette_score(data, labels)
        silhouette_scores.append(score)
        
        print(f'Gamma: {gamma}, Silhouette Score: {score}')
        
    return silhouette_scores


# In[9]:


# Define gamma values and number of clusters
gamma_values = [0.1, 0.5, 1.0, 2.0]  
n_clusters = 3  

# Run the experiment
scores = spectral_clustering_experiment(scaled_data, gamma_values, n_clusters)


# In[11]:


# Plotting the silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(gamma_values, scores, marker='o')
plt.title('Silhouette Scores for Different Gamma Values')
plt.xlabel('Gamma')
plt.ylabel('Silhouette Score')
plt.grid()
plt.show()


# In[ ]:




