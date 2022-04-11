#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.insert(0, '..')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('aimport', 'std_func')


# ## Cosine Similarity Distances
# Cosine similarity measures the similarity between two vectors of an inner product space. In text analysis, a document can be represented by its elements (words) and the frequency of each element. Comparing the frequency of words in different documents, which is the company description for companies in our case, would generate cosine similarity distance between documents. Each description generates a vector containing the frequency of each word. It measures the similarity between these companies in terms of their business description.

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df = pd.read_csv('../data/preprocessed.csv', 
                usecols = ['reportingDate', 'name', 
                           'coDescription_stopwords', 'SIC', 'SIC_desc'])


# ### Words Counting

# In[4]:


from sklearn.feature_extraction.text import CountVectorizer

Vectorizer = CountVectorizer(ngram_range = (2,4), 
                             max_features = 600)

count_data = Vectorizer.fit_transform(df['coDescription_stopwords'])


# In[5]:


wordsCount = pd.DataFrame(count_data.toarray(),columns=Vectorizer.get_feature_names())
wordsCount = wordsCount.set_index(df['name'])


# In[6]:


wordsCount


# ## Compute Cosine Similarity

# In[7]:


# Compute Cosine Similarity
from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = pd.DataFrame(cosine_similarity(wordsCount, wordsCount))
cosine_sim = cosine_sim.set_index(df['name'])
cosine_sim.columns = df['name']


# In[8]:


cosine_sim


# ## Accuracy - Confusion Matrix / ROC Curves
# ### Predictions Based on the Closest Cosine Similarity Distance

# In[9]:


cosine_sim_conf = std_func.conf_mat_cosine(cosine_sim, df)
cosine_sim_conf


# In[10]:


std_func.show_ROC_curves(df, cosine_sim_conf)


# ## Plotting

# ### Plotting on the Cosine Similarity Matrix

# In[11]:


plot_cos = std_func.pca_visualize_2d(cosine_sim, df.loc[:,["name","SIC_desc"]])


# In[12]:


std_func.pca_visualize_3d(plot_cos)


# We can see from the above 3D plot that three industries are clustered well spread, especially state commercial banks. However, real estate and software industries are closely clustered.

# We can look at the explained variance of each dimension the PCA embedding of our cosine similatiry matrix produced below:

# In[13]:


plot_cos[0].explained_variance_ratio_


# The total explained variance of the first three dimensions are:

# In[14]:


plot_cos[0].explained_variance_ratio_[0:3].sum()


# The first three dimensions explained 79% of the total variance that exists within the data.
