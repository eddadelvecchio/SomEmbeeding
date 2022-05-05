import pandas as pd
import re
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

x = ['this is very good show' , 'i had a great time on my school trip', 'such a boring movie', 'Springbreak was amazing', 'You are wrong', 'This food is so tasty', 'I had so much fun last night', 'This is crap', 'I had a bad time last month',
    'i love this product' , 'this is an amazing item', 'this food is delicious', 'I had a great time last night', 'thats right',
     'this is my favourite restaurant' , 'i love this food, its so good', 'skiing is the best sport', 'what is this', 'this product has a lot of bugs',
     'I love basketball, its very dynamic' , 'its a shame that you missed the trip', 'game last night was amazing', 'Party last night was so boring',
     'such a nice song' , 'this is the best movie ever', 'hawaii is the best place for trip','how that happened','This is my favourite band',
     'I cant believe that you did that', 'Why are you doing that, I do not gete it', 'this is tasty', 'this song is amazing']

cv = CountVectorizer(analyzer = 'word', max_features = 5000, lowercase=True, preprocessor=None, tokenizer=None, stop_words = 'english')  
vectors = cv.fit_transform(x)
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 0)
kmean_indices = kmeans.fit_predict(vectors)

pca = PCA(n_components=2)
scatter_plot_points = pca.fit_transform(vectors.toarray())

colors = ["r", "b", "c", "y", "m" ]

x_axis = [o[0] for o in scatter_plot_points]
y_axis = [o[1] for o in scatter_plot_points]
fig, ax = plt.subplots(figsize=(20,10))

ax.scatter(x_axis, y_axis, c=[colors[d] for d in kmean_indices])

for i, txt in enumerate(x):
    ax.annotate(txt, (x_axis[i], y_axis[i]))
plt.show()