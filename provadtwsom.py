import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import inspect
from sklearn.decomposition import PCA
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

vectorizer = CharVectorizer(" x.w()+-%<>/*=:_[]';T#&?")
with open("test.txt", "r") as f:
    windows = f.readlines()
windows = [x.replace("\n", "").strip() for x in windows]
windows = [y for x in windows for y in x.split(" ")]
target_length = max(len(window) for window in windows)
matrix = vectorizer.transform(windows, target_length)
sys.stdout= open("resultDictionary.txt","w")
print(matrix)

## The Multivariate DTW-SOM was designed to take input data in shape of (n, k, t)
## n is the number of data records, k is the number of variables, t is the number of timestamps
data_all_new = np.swapaxes(matrix, 1, 2)

som_shape = (6,6)
som = dtwsom.MultiDtwSom(6, 6, data_all_new.shape[2], bands =data_all_new.shape[1], w = [(1/24),(1/24),(1/24),(1/24),(1/24),(1/24),(1/24),(1/24),(1/24),(1/24),(1/24),(1/24),(1/24),(1/24),(1/24),(1/24),(1/24),(1/24),(1/24),(1/24),(1/24),(1/24),(1/24),(1/24)], sigma=1, learning_rate=0.5, random_seed=10,gl_const="sakoe_chiba", scr=60)
som.pca_weights_init(data_all_new)
som.train_batch(data_all_new, 1000, verbose=False)
weights = som.get_weights()
weights_init = som.pca_weights_init(data_all_new)
#results = som.labels_map(data_all_new,windows)
#results = np.asmatrix(results)
#results=np.swapaxes(results,2,1)
#results = [vectorizer.reverse_transform(row) for row in results]
#for result in results:
#    neuron_winner = ''.join([str(v) for v in result])
#    print(neuron_winner)

final = som.labels_map(data_all_new,windows)
print("CLUSTER WINNER COORDINATES: ")
print(final)
print(final.values)
for cnt,xx in enumerate(data_all_new):
        prova = som.winner(xx) 
        print(prova)
        print(windows[cnt])
prova2 = som._activate(data_all_new)
print(prova2)
#prova3 = som.DTW_update(data_all_new[0], data_all_new[1], windows)
#print(prova3)
#prova4 = som.activation_response(data_all_new[0])
#print(prova4)
prova5 = som.quantization_error(data_all_new)
print(prova5)
prova6 = som.win_map(data_all_new)
print(prova6)
prova7 = som.activate(data_all_new)
print(prova7)
prova8 = som.ims
print(prova8)
prova9 = som.neighborhood
print(prova9)
#prova10 = som.distance_map()

def callback(self, data_all_new):
    vector = ((np.array(weights)[0]+np.array(weights)[1]+np.array(weights)[2]+np.array(weights)[3]+np.array(weights)[4]+np.array(weights)[5]+np.array(weights)[6]+np.array(weights)[7]+np.array(weights)[8]+np.array(weights)[9]+np.array(weights)[10]+np.array(weights)[11]+np.array(weights)[12]+np.array(weights)[13]+np.array(weights)[14]+np.array(weights)[15]+np.array(weights)[16]+np.array(weights)[17]+np.array(weights)[18]+np.array(weights)[19]+np.array(weights)[20]+np.array(weights)[21]+np.array(weights)[22]+np.array(weights)[23]).reshape(36, data_all_new.shape[2]))
    print(vector)        
    w = som.winner(vector)
    return w[0], w[1]

prova11 = callback(som, data_all_new)
print(prova11)

KMeans_X = np.stack((np.array(weights)[0]+np.array(weights)[1]+np.array(weights)[2]+np.array(weights)[3]+np.array(weights)[4]+np.array(weights)[5]+np.array(weights)[6]+np.array(weights)[7]+np.array(weights)[8]+np.array(weights)[9]+np.array(weights)[10]+np.array(weights)[11]+np.array(weights)[12]+np.array(weights)[13]+np.array(weights)[14]+np.array(weights)[15]+np.array(weights)[16]+np.array(weights)[17]+np.array(weights)[18]+np.array(weights)[19]+np.array(weights)[20]+np.array(weights)[21]+np.array(weights)[22]+np.array(weights)[23]).reshape(36, data_all_new.shape[2]))
KMeans_all = KMeans(n_clusters=6, random_state=0).fit(KMeans_X)

KMeans_all.fit(data_all_new[:,:,0])
details = [(name, cluster) for name, cluster in zip(data_all_new, KMeans_all.labels_)]
max = []
for detail in details:
    max.append(detail[0])
strings=np.swapaxes(max,2,1)
print(details)
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
        frame = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
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
        ax.set_facecolor(mapper.to_rgba(KMeans_all.labels_[i*6 +j]))
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.savefig("kmeansDictionary.png", dpi=200)
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

centroids = initialize_centroids(6, data_all_new)

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
