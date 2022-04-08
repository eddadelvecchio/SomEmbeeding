## This notebook is designed to be runned under the current file directory
## You do not need to install the DTW-SOM package, since it will be imported from local file directory
## Please refer to https://github.com/Kenan-Li/dtwsom for installation
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

## The Multivariate DTW-SOM was designed to take input data in shape of (n, k, t)
## n is the number of data records, k is the number of variables, t is the number of timestamps
data_all_new = np.swapaxes(matrix, 1, 2)

som_shape = (6,6)
som = dtwsom.MultiDtwSom(6, 6, data_all_new.shape[2], bands =data_all_new.shape[1], w = [(1/27),(1/27),(1/27),(1/27),(1/27),(1/27),(1/27),(1/27),(1/27),(1/27),(1/27),(1/27),(1/27),(1/27),(1/27),(1/27),(1/27),(1/27),(1/27),(1/27),(1/27),(1/27),(1/27),(1/27),(1/27),(1/27),(1/27)], sigma=1, learning_rate=0.5, random_seed=10,gl_const="sakoe_chiba", scr=60)
som.pca_weights_init(data_all_new)
som.train_batch(data_all_new, 50, verbose=False)
weights = som.get_weights()

KMeans_X = np.stack((np.array(weights)[0]+np.array(weights)[1]+np.array(weights)[2]+np.array(weights)[3]).reshape(36, data_all_new.shape[2]))
KMeans_all = KMeans(n_clusters=10, random_state=0).fit(KMeans_X)

details = [(name, cluster) for name, cluster in zip(data_all_new, KMeans_all.labels_)]
max = []
for detail in details:
    max.append(detail[0])
strings=np.swapaxes(max,2,1)
print(details)
print(max)
strings = [vectorizer.reverse_transform(row) for row in strings]
for string in strings:
    listToStr = ''.join([str(v) for v in string])
    print(listToStr)

plt.figure(figsize=(15, 15))
norm = matplotlib.colors.Normalize(vmin=0, vmax=3, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.Set2)
for i in range(6):
    maxylim= np.max(matrix)
    for j in range(6):
        ax = plt.subplot(6, 6, i*6 + 1+j)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_ylim((0,maxylim/5))
        plt.plot(np.array(weights)[0, i, j, :], color= 'blue') 
        plt.plot(np.array(weights)[1, i, j, :], color= 'green') #a
        plt.plot(np.array(weights)[2, i, j, :], color= 'orange') #b
        plt.plot(np.array(weights)[3, i, j, :], color= 'red')    #c
        plt.plot(np.array(weights)[4, i, j, :], color= 'yellow')  #d
        plt.plot(np.array(weights)[5, i, j, :], color= 'black') #e
        plt.plot(np.array(weights)[6, i, j, :], color= 'brown') #f
        plt.plot(np.array(weights)[7, i, j, :], color= 'purple') #g
        plt.plot(np.array(weights)[8, i, j, :], color= 'olive') #h
        plt.plot(np.array(weights)[9, i, j, :], color= 'maroon') #i
        plt.plot(np.array(weights)[10, i, j, :], color= 'pink') #j
        plt.plot(np.array(weights)[11, i, j, :], color= 'lightblue') #k
        plt.plot(np.array(weights)[12, i, j, :], color= 'lightgreen') #l
        plt.plot(np.array(weights)[13, i, j, :], color= 'darkblue') #m
        plt.plot(np.array(weights)[14, i, j, :], color= 'darkgreen') #n
        plt.plot(np.array(weights)[15, i, j, :], color= 'fuchsia') #o
        plt.plot(np.array(weights)[16, i, j, :], color= 'gold') #p
        plt.plot(np.array(weights)[17, i, j, :], color= 'lime') #q
        plt.plot(np.array(weights)[18, i, j, :], color= 'cyan') #r
        plt.plot(np.array(weights)[19, i, j, :], color= 'green') #s
        plt.plot(np.array(weights)[20, i, j, :], color= 'grey') #t
        plt.plot(np.array(weights)[21, i, j, :], color= 'red') #u
        plt.plot(np.array(weights)[22, i, j, :], color= 'magenta') #v
        plt.plot(np.array(weights)[23, i, j, :], color= 'olive') #w
        plt.plot(np.array(weights)[24, i, j, :], color= 'violet') #x
        plt.plot(np.array(weights)[25, i, j, :], color= 'plum') #y
        plt.plot(np.array(weights)[26, i, j, :], color= 'darkgrey') #z
        ax.set_facecolor(mapper.to_rgba(KMeans_all.labels_[i*6 +j]))
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.savefig("kmeans.png", dpi=200)
plt.show()

def initialize_centroids(k, data_all_new):

    n_dims = data_all_new.shape[1]
    centroid_min = data_all_new.min().min()
    centroid_max = data_all_new.max().max()
    centroids = []

    for centroid in range(k):
        centroid = np.random.uniform(centroid_min, centroid_max, n_dims)
        centroids.append(centroid)

    centroids = pd.DataFrame(centroids)

    return centroids

centroids = initialize_centroids(4, data_all_new)

def calculate_error(a,b):
    error = np.square(np.sum((a-b)**2))
    return error 

errors = np.array([])
for centroid in range(centroids.shape[0]):
    error = calculate_error(centroids.iloc[centroid, :2], data_all_new.all())
    errors = np.append(errors, error)

plt.scatter(data_all_new.all(), data_all_new.all(),  marker = 'o', alpha = 0.2)
plt.scatter(centroids.iloc[:,0], centroids.iloc[:,1],  marker = 'o', c = 'r')
plt.scatter(data_all_new.all(), data_all_new.all(),  marker = 'o', c = 'g')
for i in range(centroids.shape[0]):
    plt.text(centroids.iloc[i,0]+1, centroids.iloc[i,1]+1, s = centroids.index[i], c = 'r')
plt.show()

print(dtwsom.MiniSom.win_map(data_all_new, som.winner))

