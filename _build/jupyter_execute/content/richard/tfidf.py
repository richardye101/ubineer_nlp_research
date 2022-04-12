#!/usr/bin/env python
# coding: utf-8

# # Word embedding using tf-idf matrices
# 
# This notebook uses tf-idf matrices to create word embeddings for companies. 
# 
# Tf-Idf stands for _term frequency - inverse document frequency_. Each row in this matrix represents one document (in this case, one company) and each column represents a word (or n-gram, a sequence of words like "University of Toronto"). A term frequency matrix has the count of occurences of a given word for each document, while a tf-idf matrix performs a transformation on that term frequency matrix. The computation for each cell uses:
# 
# $\sum^n_{i=1} x $
# - Where **t** is the current term we are process, and **d** is the current document we are looking in
# - Where **N** is the total number of documents in the document set and **df(t)** is the document frequency of t;
#     - The document frequency is the number of documents in the document set that contain the term t  
# ^ From [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html)
# 
# We'll be training a model from `sklearn`'s `feature_extraction` class, first using a `CountVectorizer` to obtain term-frequencies of terms of size 2-4 (we do this as some terms such as "cloud computing" carry more meaning than those words do separately. We also only select the top 600 words by freqeuncy as the columns. This is then piped into a `TfidfTransformer`, augmenting the values so the values more accurately represent the **importance** of a given term.

# In[1]:


import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '..')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('aimport', 'std_func')

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

df = pd.read_csv("../data/preprocessed.csv")
pipe = Pipeline([('count', CountVectorizer(ngram_range = (2,4),
                                           stop_words = 'english', max_features = 600)),
                  ('tfidf', TfidfTransformer())]).fit(df["coDescription_stopwords"])
tfidf =  pd.DataFrame(pipe.transform(df["coDescription_stopwords"]).toarray())
plot = std_func.pca_visualize_2d(tfidf, df.loc[:,["name","SIC_desc"]])
std_func.pca_visualize_3d(plot)

