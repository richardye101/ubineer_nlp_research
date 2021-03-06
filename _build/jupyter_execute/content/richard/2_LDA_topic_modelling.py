#!/usr/bin/env python
# coding: utf-8

# ## LDA (Latent Dirichlet Allocation)
# 
# LDA is a generative statistical model that allows sets of observations to be explained by unobserved groups that explain why some parts of the data are similar. For example, if observations are words collected into documents, it posits that each document is a mixture of a small number of topics and that each word’s presence is attributable to one of the document’s topics.
# 
# To connect this back to bag-of-words (term frequency), the former approach can be thought of as a simplistic probabilistic model of documents as distributions over words. The bag-of-words vector then represents the best approximation we have for the unnormalized distribution-of-words in each document; but document here is the basic probabilistic unit, each a single sample of its unique distribution.
# 
# 
# The crux of the matter, then, is to move from this simple probabilistic model of documents as distributions over words to a more complex one by adding a latent (hidden) intermediate layer of K topics.

# - From CSCD25, by Ashton Anderson ![image](../images/lda_cscd25.png)

# We are explaining documents (companies in our case) by their distribution across topics, which themselves are explained by a distribution of words.
# 
# ![image.png](../images/lda.jpeg)

# ## Lets get to the code!
# 
# First we need to load in the functions and data:

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


# The LDA decomposition is based off of a tf-idf matrix, which we calculated earlier. As you can see, its quite simple to create a data pipeline that passes our data through the models we want to fit.

# In[2]:


from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

pipe = Pipeline([('count', CountVectorizer(
                                           ngram_range = (2,4),
                                           stop_words = 'english', max_features = 600)),
                 ('tfidf', TfidfTransformer()),
                 ('lda', LatentDirichletAllocation(n_components = 8))]).fit(df["coDescription_stopwords"])


# Below we have the matrix of our 8 (arbitrarily) chosen topics and their vectors as they lie in our 600 term vector space:

# In[3]:


pd.DataFrame(pipe["lda"].components_)


# Below we have the top 5 terms for each topic that we've created from our corpus:

# In[4]:


lda_weights = pd.DataFrame(pipe["lda"].components_, columns = pipe["count"].get_feature_names_out())

lda_weights = lda_weights.melt(ignore_index = False).reset_index()

lda_weights.groupby('index').apply(lambda x:x.sort_values('value', ascending=False).iloc[0:5])


# Here we transform company reports using the data pipeline we built earlier which gives us a probability of belonging to each of the 8 topics:

# In[5]:


lda_df = pd.DataFrame(pipe.transform(df['coDescription']))
lda_df


# ## Plotting the results
# 
# Here are the results of running our data through LDA.

# In[6]:


plot = std_func.pca_visualize_2d(lda_df, df.loc[:,["name","SIC_desc"]])


# In[7]:


std_func.pca_visualize_3d(plot)


# You may have noticed that these plots look very... _triangular_. What these plots visualize are the 8 probability dimensions calculated using LDA and projected to this lower dimensional space. 
# 
# These embeddings are not exactly helpful in clustering documents together, but they can give us a great view into the latent topics that exist within our corpus.
# 
# If we examine the explained variance ratio, we see that the top three dimensions don't actually explain that much of the variation that exists within our data/companies. This is as expected.

# In[8]:


plot[0].explained_variance_ratio_

