from audioop import reverse
import sys
import os
from matplotlib.patches import ConnectionPatch
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance as dist
import pandas as pd
import dtwsom
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from vectorizer import CharVectorizer

vectorizer = CharVectorizer(" abcdefghijklmnopqrstuvwxyz")
with open("words_alpha.txt", "r") as f:
    windows = f.readlines()
target_length = max(len(window) for window in windows)
#matrix = vectorizer.transform(windows, target_length)
#matrix=vectorizer.sanitizer(windows, matrix)
sys.stdout= open("result.txt","w")
matrix= vectorizer.transform(windows,target_length)
print(matrix)
#strings = [vectorizer.reverse_transform(row) for row in matrix]
#for string in strings:
#    listToStr = ' '.join([str(v) for v in string])
#    print(listToStr)
#matrix_new= np.swapaxes(matrix, 0,1)

data_all_new = np.swapaxes(matrix, 1, 2)

som_shape = (10,10)
som = dtwsom.MultiDtwSom(10, 10, data_all_new.shape[2], bands =data_all_new.shape[1], w = [0.25, 0.25, 0.25, 0.25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], sigma=1, learning_rate=0.5, random_seed=10,gl_const="sakoe_chiba", scr=60)
som.pca_weights_init(data_all_new)
som.train_random(data_all_new, 200, verbose=False)
weights = som.get_weights()
num = np.swapaxes(data_all_new,2,1)
#num = [vectorizer.reverse_transform(row) for row in data_all_new]
num = [vectorizer.reverse_transform(row) for row in num]
for row in windows:
    listToStr = ''.join([str(v) for v in row])
    print(listToStr)

plt.figure(figsize=(8, 8))
wmap = {}
im = 0
for x, t in zip(data_all_new, listToStr):  # scatterplot
    w = som.winner(data_all_new)
    wmap[w] = im
    plt. text(w[0]+.5,  w[1]+.5,  str(t), fontdict={'weight': 'bold',  'size': 11})
    im = im + 1
plt.axis([0, som.get_weights().shape[0], 0,  som.get_weights().shape[1]])
plt.savefig('som_digts.png')
plt.show()


pca = PCA(n_components=1)
train_x = pd.get_dummies(listToStr)
scatter_plot_points = pca.fit_transform(train_x)

colors = ["r", "b", "c", "y", "m" ]

x_axis = [o[0] for o in scatter_plot_points]
y_axis = [o[0] for o in scatter_plot_points]
fig, ax = plt.subplots(figsize=(20,10))

#ax.scatter(x_axis, y_axis, c=[colors[d] for d in w])

#for i, txt in enumerate(data_all_new):
#    ax.annotate(txt, (x_axis[i], y_axis[i]))

def get_cluster_words(cluster_index, num, cv):
    cluster_words = []
    for i in range(len(cluster_index)):
        if cluster_index[i] == cluster_index[i-1]:
            cluster_words.append( num.get_feature_names()[i])
    return cluster_words

print(get_cluster_words(w, num, data_all_new))
plt.show()



#print(matrix_new.shape)