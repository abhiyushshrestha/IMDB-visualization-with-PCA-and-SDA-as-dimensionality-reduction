#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 13:28:28 2018

@author: abhiyush
"""

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# Custom library
from preprocessingPCA import PreprocessDataSets
from vectorizer import Vectorizer_tfidf

corpus, y = PreprocessDataSets(Path = "/home/abhiyush/mPercept/Natural Language Processing/Sentiment Analysis/IMDB reviews/imdb_reviews.csv")

x_train, x_test, y_train, y_test = Vectorizer_tfidf(corpus, y)

#Applying PCA to reduce the dimensionality 

#Using PCA with 2 components
pca = PCA(n_components = 2)
x_train_pca_2d = pca.fit_transform(x_train)
explained_variance_ratio = pca.explained_variance_ratio_
explained_variance_pca = pca.explained_variance_


datasets_pca = pd.DataFrame(x_train_pca_2d)
datasets_pca['Likes'] = y_train

# Creating a separate list for positive and negative reviews 

likes_class_pca_neg = []
likes_class_pca_pos = []

for i in range(0, len(datasets_pca)):
    if (datasets_pca.iloc[i, 2] == 0):
        likes_class_pca_neg.append(datasets_pca.iloc[i, 0:2].values)
    else:
        likes_class_pca_pos.append(datasets_pca.iloc[i, 0:2].values)

likes_class_pca_neg_df = pd.DataFrame(likes_class_pca_neg)
likes_class_pca_pos_df = pd.DataFrame(likes_class_pca_pos)

Xp_pca_2d = likes_class_pca_pos_df.iloc[:,0]
Yp_pca_2d = likes_class_pca_pos_df.iloc[:,1]

Xn_pca_2d = likes_class_pca_neg_df.iloc[:,0]
Yn_pca_2d = likes_class_pca_neg_df.iloc[:,1]

# Visualizing the datasets in 2D
fig, ax = plt.subplots(figsize = (10,10))
ax.scatter(Xn_pca_2d, Yn_pca_2d, c = 'r')
ax.scatter(Xp_pca_2d, Yp_pca_2d, c = 'b')


# Using pca (Singular Value Decomposition) with 3 components

pca_3d = PCA(n_components = 3)
x_train_pca_3d = pca_3d.fit_transform(x_train)
x_test_pca_3d = pca_3d.transform(x_test)
explained_variance_3d = pca_3d.explained_variance_ratio_


datasets_pca_3d = pd.DataFrame(x_train_pca_3d)
datasets_pca_3d['Likes'] = y_train

#Making an empty list 
likes_class_pca_neg_3d = []
likes_class_pca_pos_3d = []


# Using a for loop to create a separate list for positive and negative likes
for i in range(0, len(datasets_pca_3d)):
    if (datasets_pca_3d.iloc[i, 3] == 0):
        likes_class_pca_neg_3d.append(datasets_pca_3d.iloc[i, 0:3].values)
    else:
        likes_class_pca_pos_3d.append(datasets_pca_3d.iloc[i, 0:3].values)

likes_class_pca_neg_df_3d = pd.DataFrame(likes_class_pca_neg_3d)
likes_class_pca_pos_df_3d = pd.DataFrame(likes_class_pca_pos_3d)

Xp_pca_3d = likes_class_pca_pos_df_3d.iloc[:,0]
Yp_pca_3d = likes_class_pca_pos_df_3d.iloc[:,1]
Zp_pca_3d = likes_class_pca_pos_df_3d.iloc[:,2]

Xn_pca_3d = likes_class_pca_neg_df_3d.iloc[:,0]
Yn_pca_3d = likes_class_pca_neg_df_3d.iloc[:,1]
Zn_pca_3d = likes_class_pca_neg_df_3d.iloc[:,2]

fig, ax = plt.subplots(figsize = (10,10))
ax = Axes3D(fig)

ax.scatter(Xp_pca_3d, Yp_pca_3d, Zp_pca_3d, c = 'r', s = 150)
ax.scatter(Xn_pca_3d, Yn_pca_3d, Zn_pca_3d, c = 'b', s = 150)
