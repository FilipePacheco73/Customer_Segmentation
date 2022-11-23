# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 19:44:13 2022

@author: Filipe Pacheco

Customer Segmentation utilizing K-Means

Original dataset available in https://www.kaggle.com/datasets/iamsouravbanerjee/customer-segmentation-51k-records

Theorical base from https://neptune.ai/blog/customer-segmentation-using-machine-learning

"""

# Imports - Libraries

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = pd.read_csv('Customer Segmentation.csv')

# Discover quantity of unique values

unique_values = []
for i in data.columns.tolist(): 
    print('Number of unique values:',len(pd.unique(data[i])),'of column',i)
    unique_values.append([i,len(pd.unique(data[i]))])
    
# Order by unique values aiming to select only the 5 least different values per category 
unique_values = pd.DataFrame(unique_values,columns=['column','value']).sort_values(['value'])

selected_columns = unique_values['column'][:5].tolist()

# Reduce dataset

data_selected = data[selected_columns]

# Encoder categorical variables into numbers

enc = OrdinalEncoder()
enc.fit(data_selected)
enc.categories_ # check the categories

data_encoded = enc.transform(data_selected) # transform data

print("\n",data_encoded) # check transformed data


# Create the K means model for different values of K - Objective: find the best number of K - elbow method

def try_different_clusters(K, data):

    cluster_values = list(range(1, K+1))
    inertias=[]

    for c in cluster_values:
        model = KMeans(n_clusters = c,init='k-means++',max_iter=400,random_state=42)
        model.fit(data)
        inertias.append(model.inertia_)

    return inertias

# Find output for k values between 1 to 12 
outputs = try_different_clusters(12, data_encoded)
distances = pd.DataFrame({"clusters": list(range(1, 13)),"sum of squared distances": outputs})

# Finding optimal number of clusters k - Graphical and mathematical

fig, ax = plt.subplots()
ax.plot(distances['clusters'], distances['sum of squared distances'])

ax.set(xlabel='Number of clusters', ylabel='Sum of squared distances',title='Finding optimal number of clusters using elbow method')
ax.grid()

fig.savefig("kmeans_elbow_method.png")
plt.show()


sum_distances_acc = 0
cont = 0
while sum_distances_acc < abs(.9*np.sum(np.diff(distances['sum of squared distances']))): # sum of all distances
    sum_distances_acc += abs(np.diff(distances['sum of squared distances'])[cont])
    cont += 1

# Create K-means models

kmeans_model = KMeans(n_clusters = cont, init='k-means++',  max_iter=400, random_state=42)

# Train the model and check the segmented data

kmeans_model.fit_predict(data_encoded)

data_selected['clusters'] = kmeans_model.labels_