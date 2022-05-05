import numpy as np
from pyparsing import line
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering, linkage_tree
import distance
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from itertools import cycle
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from scipy.cluster.hierarchy import dendrogram, linkage

f = open("test.txt", "r")    
words =[line.rstrip('\n') for line in f] 
words = np.asarray(words)
#calcolo dell'EDIT-DISTANCE tra due parole del vocabolario trasformate in n p arrays, normalizzato il risultato (-1*), uso il risultato come classificatore per l'affinity propagation
lev_similarity = -1*np.array([[distance.levenshtein(w1,w2) for w1 in words] for w2 in words])

aggcl = AgglomerativeClustering(affinity="euclidean", n_clusters=34, linkage="ward", compute_full_tree=True)
#addestro il modello usando le features della EDIT_DISTANCE come matrice di similarità tra coppie di parole
aggcl.fit(lev_similarity)
#estraggo i cluster
cluster_centers_indices = aggcl.labels_
for cluster_id in np.unique(aggcl.labels_):
    exemplar = words[aggcl.labels_[cluster_id]]
    cluster = np.unique(words[np.nonzero(aggcl.labels_==cluster_id)])
    cluster_str = ", ".join(cluster)
    print(" - CLUSTER WINNER: %s* - CLUSTER WORDS: %s" % (exemplar, cluster_str))

colors = cycle("bgrcmykbgrcmykbgrcmykbgrcmyk")
for k, col in zip(range(aggcl.n_clusters), colors):
    class_members = aggcl.labels_ == k
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

plt.title("Estimated number of clusters: %d" % aggcl.n_clusters)
plt.show()
