#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 13:48:40 2018

@author: abhiyush
"""

import pandas as pd
import re

from nltk.stem import WordNetLemmatizer


def PreprocessDataSets(Path):
    datasets = pd.read_csv(Path, delimiter = '\t', quoting = 3, header = None)
    
    # Naming the columns
    datasets.columns = ['reviews', 'likes']
    y = datasets.iloc[:,1].values

    #Initializing the object for lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    #Preprocessing the datasets 
    corpus = []
    for i in range(0, len(datasets)):
        review = re.sub('[^a-zA-Z]', ' ', datasets['reviews'][i])
        review = review.lower()
        review = review.split()
        review = [lemmatizer.lemmatize(word) for word in review]
        review = ' '.join(review)
        corpus.append(review)
    return corpus,y