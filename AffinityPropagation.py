import numpy as np
import sys
from sklearn.cluster import AffinityPropagation
import distance
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from itertools import cycle
from scipy.cluster.hierarchy import dendrogram, linkage
from weighted_levenshtein import lev, osa, dam_lev

f = open("test.txt", "r")    
words =[line.rstrip('\n') for line in f] 
words = np.asarray(words)
#calcolo dell'EDIT-DISTANCE tra due parole del vocabolario trasformate in n p arrays, normalizzato il risultato (-1*), uso il risultato come classificatore per l'affinity propagation

lev_similarity = -1*np.array([[distance.levenshtein(w1,w2) for w1 in words] for w2 in words])

affprop = AffinityPropagation(affinity="euclidean", damping=0.5)
#addestro il modello usando le features della EDIT_DISTANCE come matrice di similarità tra coppie di parole
affprop.fit(lev_similarity)
#estraggo i cluster
cluster_centers_indices = affprop.cluster_centers_indices_
n_clusters_ = len(cluster_centers_indices)
for cluster_id in np.unique(affprop.labels_):
    exemplar = words[affprop.cluster_centers_indices_[cluster_id]]
    cluster = np.unique(words[np.nonzero(affprop.labels_==cluster_id)])
    cluster_str = ", ".join(cluster)
    print(" - CLUSTER WINNER: %s* - CLUSTER WORDS: %s" % (exemplar, cluster_str))

colors = cycle("bgrcmykbgrcmykbgrcmykbgrcmyk")
for k, col in zip(range(n_clusters_), colors):
    class_members = affprop.labels_ == k
    cluster_center = words[cluster_centers_indices[k]]
    plt.plot(words[class_members], words[class_members], col + ".")
    #rendo il grafico attivo prendendendo le coordinate di ogni parola indicizzata sugli assi
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.plot(
        cluster_center[0],
        #cluster_center[1],
        "o",
        markerfacecolor=col,
        markeredgecolor="k",
        markersize=6,
    )
    for x in words[class_members]:
       plt.plot([cluster_center[0]], #[cluster_center[1]],
        col)

plt.title("Estimated number of clusters: %d" % n_clusters_)
plt.savefig("affprop.png", dpi=200)
plt.show()

linked = linkage(lev_similarity, 'single')

plt.figure(figsize=(10, 7))
frame = plt.gca()
frame.axes.get_xaxis().set_visible(True)
frame.axes.get_yaxis().set_visible(True)
dendrogram(linked,
            orientation='top',
            labels=words,
            distance_sort='descending',
            show_leaf_counts=True,
            truncate_mode = 'level'),
plt.savefig("dendrogram.png", dpi=200)         
plt.show()
