import pandas as pd
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
import dtwsom
import pickle
from sklearn.cluster import KMeans
from vectorizer import CharVectorizer
from array import array
import plotly.graph_objects as go
import seaborn as sn
from matplotlib.gridspec import GridSpec

vectorizer = CharVectorizer(" abcdefghijklmnopqrstuvwxyz")
with open("words_alpha.txt", "r") as f:
    windows = f.readlines()
target_length = max(len(window) for window in windows)
matrix = vectorizer.transform(windows, target_length)
sys.stdout= open("result.txt","w")
print(matrix)
data_all_new = np.swapaxes(matrix, 1, 2)

som_shape = (10,10)
som = dtwsom.MultiDtwSom(10, 10, data_all_new.shape[2], bands =data_all_new.shape[1], w = [0.25, 0.25, 0.25, 0.25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], sigma=1, learning_rate=0.5, random_seed=10,gl_const="sakoe_chiba", scr=60)
som.pca_weights_init(data_all_new)
som.train_random(data_all_new, 20, verbose=True)
weights = som.get_weights()

winner_coordinates = np.array([som.winner(x) for x in data_all_new]).T
cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)
print(winner_coordinates)

def print_cluster_winner_element(cluster_index, data_all_new):
    cluster_index = np.array(cluster_index)
    cluster_index = cluster_index.reshape(som_shape)
    for i in range(som_shape[0]):
        for j in range(som_shape[1]):
            print(vectorizer.reverse_transform(data_all_new[cluster_index[i,j]]))


#details = [(name, cluster) for name, cluster in (data_all_new, labels)]
#strings=np.swapaxes(details,0,1)
#for string in strings:
#    listToStr = ' '.join([str(v) for v in string])
#    listToStr
#print(listToStr)

for c in np.unique(cluster_index):
    plt.scatter(data_all_new[cluster_index == c, 0],
                data_all_new[cluster_index == c, 1], label='cluster='+str(c), alpha=.7)

for centroid in som.get_weights():
    plt.scatter(centroid[:, 0], centroid[:, 1], marker='x', 
                s=50, linewidths=1, color='k', label='centroid')


                
plt.legend();
plt.show()









