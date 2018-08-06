#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 15:40:40 2018

@author: abhiyush
"""

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import TruncatedSVD
from mpl_toolkits.mplot3d import Axes3D

# Custom library
from preprocessingPCA import PreprocessDataSets
from vectorizer import Vectorizer_tfidf

corpus, y = PreprocessDataSets(Path = "/home/abhiyush/mPercept/Natural Language Processing/Sentiment Analysis/IMDB reviews/imdb_reviews.csv")

x_train, x_test, y_train, y_test = Vectorizer_tfidf(corpus, y)

#Applying svd to reduce the dimensionality 

#Using svd with 2 components
svd = TruncatedSVD(n_components = 2)
x_train_svd_2d = svd.fit_transform(x_train)
explained_variance_ratio = svd.explained_variance_ratio_
explained_variance_svd = svd.explained_variance_


datasets_svd = pd.DataFrame(x_train_svd_2d)
datasets_svd['Likes'] = y_train

# Creating a separate list for positive and negative reviews 

likes_class_svd_neg = []
likes_class_svd_pos = []

for i in range(0, len(datasets_svd)):
    if (datasets_svd.iloc[i, 2] == 0):
        likes_class_svd_neg.append(datasets_svd.iloc[i, 0:2].values)
    else:
        likes_class_svd_pos.append(datasets_svd.iloc[i, 0:2].values)

likes_class_svd_neg_df = pd.DataFrame(likes_class_svd_neg)
likes_class_svd_pos_df = pd.DataFrame(likes_class_svd_pos)

Xp_svd_2d = likes_class_svd_pos_df.iloc[:,0]
Yp_svd_2d = likes_class_svd_pos_df.iloc[:,1]

Xn_svd_2d = likes_class_svd_neg_df.iloc[:,0]
Yn_svd_2d = likes_class_svd_neg_df.iloc[:,1]

# Visualizing the datasets in 2D
fig, ax = plt.subplots(figsize = (10,10))
ax.scatter(Xn_svd_2d, Yn_svd_2d, c = 'r')
ax.scatter(Xp_svd_2d, Yp_svd_2d, c = 'b')


# Using svd (Singular Value Decomposition) with 3 components

svd_3d = TruncatedSVD(n_components = 3)
x_train_svd_3d = svd_3d.fit_transform(x_train)
x_test_svd_3d = svd_3d.transform(x_test)
explained_variance_3d = svd_3d.explained_variance_ratio_


datasets_svd_3d = pd.DataFrame(x_train_svd_3d)
datasets_svd_3d['Likes'] = y_train

#Making an empty list 
likes_class_svd_neg_3d = []
likes_class_svd_pos_3d = []


# Using a for loop to create a separate list for positive and negative likes
for i in range(0, len(datasets_svd_3d)):
    if (datasets_svd_3d.iloc[i, 3] == 0):
        likes_class_svd_neg_3d.append(datasets_svd_3d.iloc[i, 0:3].values)
    else:
        likes_class_svd_pos_3d.append(datasets_svd_3d.iloc[i, 0:3].values)

likes_class_svd_neg_df_3d = pd.DataFrame(likes_class_svd_neg_3d)
likes_class_svd_pos_df_3d = pd.DataFrame(likes_class_svd_pos_3d)

Xp_svd_3d = likes_class_svd_pos_df_3d.iloc[:,0]
Yp_svd_3d = likes_class_svd_pos_df_3d.iloc[:,1]
Zp_svd_3d = likes_class_svd_pos_df_3d.iloc[:,2]

Xn_svd_3d = likes_class_svd_neg_df_3d.iloc[:,0]
Yn_svd_3d = likes_class_svd_neg_df_3d.iloc[:,1]
Zn_svd_3d = likes_class_svd_neg_df_3d.iloc[:,2]

fig, ax = plt.subplots(figsize = (10,10))
ax = Axes3D(fig)

ax.scatter(Xp_svd_3d, Yp_svd_3d, Zp_svd_3d, c = 'r', s = 150)
ax.scatter(Xn_svd_3d, Yn_svd_3d, Zn_svd_3d, c = 'b', s = 150)
