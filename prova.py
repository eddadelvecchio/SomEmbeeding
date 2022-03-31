import sys
import os
from matplotlib.patches import ConnectionPatch
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance as dist
import pandas as pd
import dtwsom
from sklearn.cluster import KMeans
from vectorizer import CharVectorizer

vectorizer = CharVectorizer("abcdefghijklmnopqrstuvwxyz")
with open("words_alpha.txt", "r") as f:
    windows = f.readlines()
target_length = max(len(window) for window in windows)
matrix = vectorizer.transform(windows, target_length)
#matrix=vectorizer.sanitizer(windows, matrix)
sys.stdout= open("result.txt","w")
print(matrix)
#matrix_new= np.swapaxes(matrix, 0,1)



#print(matrix_new.shape)