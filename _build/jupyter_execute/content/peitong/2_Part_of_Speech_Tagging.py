#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.insert(0, '..')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('aimport', 'std_func')


# ## Part-of-Speech (POS) Tagging
# Part-of-speech (POS) tagging is a process of grammatical classification to classify texts into list of tuples where each word in the sentence gets a tag (label) that tells its part of speech (e.g. noun, pronoun, verb, adjective, adverb). According to Asoka Diggs, a Data Scientist at Intel, his research shows that nouns are better than n-grams. As a result, we used POS tagging to extract only nouns. We have examined the case with multiple-gram nouns. However, the results do not show distinct difference between documents, which may be caused by overfitting the model. Here we only consider 1-gram nouns. We conduct the consine similarity measure on the word counts from POS tagging.
# 
# https://databricks.com/session/nouns-are-better-than-n-grams

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string


# In[3]:


df = pd.read_csv('../data/preprocessed.csv',
                 usecols = ['reportingDate', 'name', 'CIK', 'coDescription_lemmatized',
                           'coDescription_stopwords', 'SIC', 'SIC_desc'])
df = df.set_index(df.name)


# ### Word Counts from POS tagging

# In[4]:


import pattern
import collections
from pattern.en import parsetree, singularize


# In[5]:


def extract_nouns(t):
    tree = parsetree(t)
    nouns = []
    for sentence in tree:
        for word in sentence:
            if 'NN' in word.type:
                nouns.append(singularize(word.string))
    return " ".join(nouns)


# In[6]:


import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize


# In[7]:


def remove_stopwords(x):
    stop_words = set(stopwords.words('english'))

    word_tokens = word_tokenize(x)

    filtered_sentence = ' '.join([w for w in word_tokens if not w.lower() in stop_words and w.isalnum()])

    return(filtered_sentence)


# In[8]:


df['coDescription_lemmatized'][0:1].apply(extract_nouns)


# In[9]:


t = df['coDescription_lemmatized'][1:3].apply(extract_nouns).apply(remove_stopwords)
type(t)


# In[10]:


pos_desc = df['coDescription_lemmatized'].apply(extract_nouns).apply(remove_stopwords)
df['coDescription_pos'] = pos_desc
df['coDescription_pos'].head()


# ### Cosine Similarity Distance on on 1-Gram Nouns

# In[11]:


from sklearn.feature_extraction.text import CountVectorizer

Vectorizer = CountVectorizer(ngram_range = (1,1), 
                             max_features = 600)

count_data = Vectorizer.fit_transform(df['coDescription_pos'])
wordsCount_pos_tag = pd.DataFrame(count_data.toarray(),columns=Vectorizer.get_feature_names())
wordsCount_pos_tag = wordsCount_pos_tag.set_index(df['name'])
wordsCount_pos_tag


# In[12]:


# Compute Cosine Similarity
from sklearn.metrics.pairwise import cosine_similarity
cosine_sim_pos_tag = pd.DataFrame(cosine_similarity(wordsCount_pos_tag, wordsCount_pos_tag))
cosine_sim_pos_tag = cosine_sim_pos_tag.set_index(df['name'])
cosine_sim_pos_tag.columns = df['name']
cosine_sim_pos_tag


# ### Predictions Based on the Closest Cosine Similarity Distance

# In[13]:


classification = cosine_sim_pos_tag.copy(deep=True)
# set the diagonals to 0
def set_diag_zero(matrix):
    for i in range(len(matrix)):
        matrix.iloc[i,i] = 0
set_diag_zero(classification)


# In[14]:


classification.index = df["SIC_desc"]
classification.columns = df["SIC_desc"]


# In[15]:


prediction = pd.DataFrame(classification.idxmax(axis=1))
prediction.reset_index(level = 0, inplace = True)
prediction.columns = ['SIC_desc', 'SIC_desc_pred']
print("Percentage of correct predictions: ")
np.sum(np.where(prediction.iloc[:,1] == prediction.iloc[:,0], 1, 0))/len(prediction)


# ### Accuracy - Confusion Matrix / ROC Curves
# #### Predictions Based on the Closest Cosine Similarity Distance

# In[16]:


std_func.conf_mat_cosine(cosine_sim_pos_tag, df)


# In[17]:


plot_cos_pos_tag = std_func.pca_visualize_2d(cosine_sim_pos_tag, df.loc[:,["name","SIC_desc"]])


# In[18]:


std_func.pca_visualize_3d(plot_cos_pos_tag)


# In[19]:


plot_cos_pos_tag[0].explained_variance_ratio_


# In[20]:


plot_cos_pos_tag[0].explained_variance_ratio_[0:3].sum()

