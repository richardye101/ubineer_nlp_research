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


# ## Part-of-Speech (POS) Tagging - Cosine Similarity Analysis
# Part-of-speech (POS) tagging is a process of grammatical classification to classify texts into list of tuples where each word in the sentence gets a tag (label) that tells its part of speech (e.g. noun, pronoun, verb, adjective, adverb). According to Asoka Diggs, a Data Scientist at Intel, his research shows that nouns are better than n-grams. As a result, we used POS tagging to extract only nouns. We have examined the case with multiple-gram nouns. However, the results do not show distinct difference between documents, which may be caused by overfitting the model. Here we only consider 1-gram nouns. We conduct the consine similarity measure on the word counts from POS tagging.
# 
# Source: https://databricks.com/session/nouns-are-better-than-n-grams

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string


# ### Cosine Similarity Analysis

# #### Word Counts from POS tagging
# We select the words with the word type `Noun` and use `CountVectorizer` from `sklearn.feature_extraction.text` to count the term frequency for each 1-gram noun and select the top 600 nouns by frequency

# In[3]:


df = pd.read_csv('data/nouns_only.csv',
                 usecols = ['reportingDate', 'name', 'CIK', 'coDescription_lemmatized',
                           'coDescription_stopwords', 'coDescription_pos', 'SIC', 'SIC_desc'])
df = df.set_index(df.name)


# In[4]:


import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize


# In[5]:


from sklearn.feature_extraction.text import CountVectorizer

Vectorizer = CountVectorizer(ngram_range = (1,1), 
                             max_features = 600)

count_data = Vectorizer.fit_transform(df['coDescription_pos'])
wordsCount_pos_tag = pd.DataFrame(count_data.toarray(),columns=Vectorizer.get_feature_names())
wordsCount_pos_tag = wordsCount_pos_tag.set_index(df['name'])
wordsCount_pos_tag


# #### Cosine Similarity Computation on on 1-Gram Nouns
# To determine the similarity of each company's business description, we use cosine similarity analysis on this POS-tagging with only nouns embeddings.

# In[6]:


# Compute Cosine Similarity
from sklearn.metrics.pairwise import cosine_similarity
cosine_sim_pos_tag = pd.DataFrame(cosine_similarity(wordsCount_pos_tag, wordsCount_pos_tag))
cosine_sim_pos_tag = cosine_sim_pos_tag.set_index(df['name'])
cosine_sim_pos_tag.columns = df['name']
cosine_sim_pos_tag


# ### Performance Evaluation
# #### Predictions Based on the Closest Cosine Similarity Distance
# We use the closest neighborhood in terms of cosine similarity distances to evaluate the accuracy of the SIC classfication generated using POS-tagging with only 1-gram nouns embeddings and cosine similarity distances.

# In[7]:


prediction, accuracy, cm = std_func.get_accuracy(cosine_sim_pos_tag, df)


# In[8]:


cosine_sim_pos_tag_conf = std_func.conf_mat_cosine(cosine_sim_pos_tag, df)
cosine_sim_pos_tag_conf


# From the above confusion matrix, cosine similarity analysis on POS-tagging with only 1-gram nouns embeddings gives an accuray of 94% on average. For industries `Crude Petroleum and Natural Gas`, `Real Estate Investment Trusts` and `State Commercial Banks (commercial banking)`, the accuracy is above 95%. `Prepackaged Software` gives the lowest accuracy at 86%. However, this confusion matrix gives extremely high prediction, we then look into the 2-D and 3-D plots to see if they are well-clustered.

# ### Plotting
# #### Plotting on the Cosine Similarity Matrix
# We use PCA to automatically perform dimensionality reduction. First, we have a 2-D plot on cosine similarity matrix.

# In[9]:


plot_cos_pos_tag = std_func.pca_visualize_2d(cosine_sim_pos_tag, df.loc[:,["name","SIC_desc"]])


# Here we have a 3-D plot with the first three dimensions which maximize the most variance.

# In[10]:


std_func.pca_visualize_3d(plot_cos_pos_tag)


# We can see from the above 3D plot that three industries are not well clustered. `Pharmaceutical Preparations` and `State Commercial Banks (commercial banking)` seem to be more spread out than others. The other three industries
# `Crude Petroleum and Natural Gas`, `Real Estate Investment Trusts` and `Prepackaged Software` are closely clustered with each other.

# We can look at the explained variance of each dimension the PCA embedding of our cosine similatiry matrix generated from POS-tagging with only 1-gram nouns embeddings produced below:

# In[11]:


plot_cos_pos_tag[0].explained_variance_ratio_


# The total explained variance of the first three dimensions are:

# In[12]:


plot_cos_pos_tag[0].explained_variance_ratio_[0:3].sum()


# The first three dimensions explained 78% of the total variance that exists within the data.

# ### Conclusion Reporting

# In[13]:


from sklearn.metrics import classification_report
print(classification_report(prediction["y_true"], prediction["y_pred"], target_names=df["SIC_desc"].unique()))


# We can see from the above classification_report, we can conclude that cosine similarity analysis on POS-tagging with 1-gram nouns embeddings gives a good result on SIC classfication, specifically on the industries `Crude Petroleum and Natural Gas`, `Real Estate Investment Trusts` and `State Commercial Banks (commercial banking)`.
