#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.insert(0, '..')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('aimport', 'std_func')

# Hide warnings
import warnings
warnings.filterwarnings("ignore")


# ## N-grams Embeddings - Cosine Similarity Analysis
# Next, we look into cosine similarity distances to measure the descripiton similarity between companies. In this notebook, we simply use n-grams embeddings for consine similarity analysis. 
# 
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


# ### Cosine Similarity Analysis

# #### Words Counting
# For this cosine similarity analysis, we generate sequences of 2 to 4 words as one term and only select the top 600 terms by frequency.

# In[4]:


from sklearn.feature_extraction.text import CountVectorizer

Vectorizer = CountVectorizer(ngram_range = (2,4), 
                             max_features = 600)

count_data = Vectorizer.fit_transform(df['coDescription_stopwords'])


# In[5]:


wordsCount = pd.DataFrame(count_data.toarray(),columns=Vectorizer.get_feature_names())
wordsCount = wordsCount.set_index(df['name'])


# Here is the n-grams embedding matrix with the 600 2-to-4 grams as columns and the 675 companies as rows.

# In[6]:


wordsCount


# #### Cosine Similarity Computation
# Now we take in the 2-to-4 grams embeddings to analyze the text similarity. 

# In[7]:


# Compute Cosine Similarity
from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = pd.DataFrame(cosine_similarity(wordsCount, wordsCount))
cosine_sim = cosine_sim.set_index(df['name'])
cosine_sim.columns = df['name']


# The description similarity between companies range from 0 to 1. The higher the cosine similarity score, the more similar they are. 

# In[8]:


cosine_sim


# ### Performance Evaluation
# #### Predictions Based on the Closest Cosine Similarity Distance
# We use the closest neighborhood in terms of cosine similarity distances to evaluate the accuracy of the SIC classfication generated using 2-to-4 grams embeddings and cosine similarity distances.

# In[9]:


prediction, accuracy, cm = std_func.get_accuracy(cosine_sim, df)


# In[10]:


cosine_sim_conf = std_func.conf_mat_cosine(cosine_sim, df)
cosine_sim_conf


# We can see from the above confusion matrix that cosine similarity analysis on 2-to-4 grams embeddings give an accuray of 89% on average. For industries `Crude Petroleum and Natural Gas`, `Real Estate Investment Trusts` and `State Commercial Banks (commercial banking)`, the accuracy is above 90%. `Pharmaceutical Preparations` gives the lowest accuracy at 76%.

# ### Plotting

# #### Plotting on the Cosine Similarity Matrix
# We use PCA to automatically perform dimensionality reduction. First, we have a 2-D plot on cosine similarity matrix.

# In[11]:


plot_cos = std_func.pca_visualize_2d(cosine_sim, df.loc[:,["name","SIC_desc"]])


# Here we have a 3-D plot with the first three dimensions which maximize the most variance.

# In[12]:


std_func.pca_visualize_3d(plot_cos)


# We can see from the above 3D plot that three industries are clustered well spread, especially state commercial banks. However, prepackaged software industry is closely clustered with the others.

# We can look at the explained variance of each dimension the PCA embedding of our cosine similatiry matrix produced below:

# In[13]:


plot_cos[0].explained_variance_ratio_


# The total explained variance of the first three dimensions are:

# In[14]:


plot_cos[0].explained_variance_ratio_[0:3].sum()


# The first three dimensions explained 79% of the total variance that exists within the data.

# ### Conclusion Reporting

# In[15]:


from sklearn.metrics import classification_report
print(classification_report(prediction["y_true"], prediction["y_pred"], target_names=df["SIC_desc"].unique()))


# We can see from the above classification_report, we can conclude that cosine similarity analysis on 2-to-4 grams embeddings gives a good result on SIC classfication, specifically on the industries `Crude Petroleum and Natural Gas`, `Real Estate Investment Trusts` and `State Commercial Banks (commercial banking)`.
