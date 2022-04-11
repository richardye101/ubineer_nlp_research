#!/usr/bin/env python
# coding: utf-8

# # Experimenting with lemmatization SVD/LSA
# 
# LSA attempts to reduce the high dimensional data created from tf-idf into a lower dimensional space using SVD. SVD stands for **Singular Value Decomposition**. It is used commonly in conjunction with tf-idf matrices to perform what is known as **Latent Semantic Analysis (LSA)**. Contrary to PCA, SVD will not center the data before reducing dimensions, which makes it work better with sparse matrices (exactly what we have).
# 
# Otherwise, this is exactly the same as LDA Topic modelling.

# This dimensionality reduction can be performed using truncated SVD. SVD, or singular value decomposition, is a technique in linear algebra that factorizes any matrix M into the product of 3 separate matrices: 
# 
# $$M=U*S*V$$
# 
# Where S is a diagonal matrix of the singular values of M. Critically, truncated SVD reduces dimensionality by selecting only the t largest singular values, and only keeping the first t columns of U and V. In this case, t is a hyperparameter we can select and adjust to reflect the number of topics we want to find.
# 
# ![image.png](../images/lsa.png)

# In[1]:


import os
import json
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '..')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('aimport', 'std_func')

df = pd.read_csv("../data/preprocessed.csv")


# In[2]:


from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

n_components = 8
pipe = Pipeline([('count', CountVectorizer(ngram_range = (2,4),
                                           stop_words = 'english', max_features = 600)),
                 ('tfidf', TfidfTransformer())]).fit(df["coDescription_stopwords"])
#                  ('svd', TruncatedSVD(n_components = n_components))]).fit(df["coDescription"])
tf_idf = pd.DataFrame(pipe.transform(df["coDescription_stopwords"]).toarray())


# In[3]:


plot_svd = std_func.visualize_svd(tf_idf, df.loc[:,["name","SIC_desc"]])


# In[4]:


std_func.pca_visualize_3d(plot_svd)


# From the explained variance ratio, we see that the top three dimensions don't actually explain that much of the variation that exists within our data/companies.

# In[5]:


plot_svd[0].explained_variance_ratio_

