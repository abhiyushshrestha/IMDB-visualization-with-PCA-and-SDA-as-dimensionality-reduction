#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 14:13:48 2018

@author: abhiyush
"""

from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessingPCA import PreprocessDataSets

#corpus, y = PreprocessDataSets(Path = "/home/abhiyush/mPercept/Natural Language Processing/Sentiment Analysis/IMDB reviews/imdb_reviews.csv")

def Vectorizer_tfidf(corpus, y):

    #Splitting the data sets to train and test sets 
    x_train, x_test, y_train, y_test = train_test_split(corpus, y, test_size = 0.2, random_state = 0)
    
    # Using tf-idf approach to create a matrix for the reviews
    tfidf_vectorizer = TfidfVectorizer()
    fit_tfidf_vectorizer = tfidf_vectorizer.fit(x_train)
    print(fit_tfidf_vectorizer.get_feature_names())
    
    transform_tfidf_vectorizer = tfidf_vectorizer.transform(x_train)
    transform_tfidf_vectorizer_toarray = transform_tfidf_vectorizer.toarray()
    
    x_train = transform_tfidf_vectorizer_toarray
    x_test = tfidf_vectorizer.transform(x_test).toarray()
    
    x_train.shape
    x_test.shape
    
    return x_train, x_test, y_train, y_test
