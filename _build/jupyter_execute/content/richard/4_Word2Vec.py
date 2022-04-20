#!/usr/bin/env python
# coding: utf-8

# # Word2vec
# 
# Another method to find company embeddings is to use word2vec. How it works and how we aim to use is explained below.
# 
# ## How it works
# 
# Imagine I have two sentences:
# - "Formula One driver Lewis Hamilton is a seven time world champion". 
# - "Ferrari driver Sebastian Vettel fails to qualify for the fifth Grand Prix in a row"
# 
# Say I want find words that are semantically related to Lewis.
# 
# <!-- We can represent "Lewis" with [1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0], where the first 11 words are from the first sentence, and the last 14 words are from the second sentence. -->
# 
# In word2vec we can use two different algorithms, continuous bag of words (CBOW) and skip-gram negative sampling. 
# 
# ![image.png](../images/w2v_1.png)
# 
# ### Continuous bag of words (CBOW)
# 
# We take a window of n words surrounding "Lewis" and use those words as input into a neural network, using the logic that if certains words appear together often, then they're most likely semantically related. We take these surrounding words and plug them into a neural network to train weights which predict "Lewis".
# 
# We plug in a one hot vector for each word in the window and train the hidden layer to output probabilities of the current word ("Lewis"). Therefore the order of the words do not matter, just what the words are.
# 
# ![image.png](../images/w2v_2.png)
# 
# ### Skip Gram Negative Sampling
# 
# Whereas, in the second option of using the continuous skip-gram architecture; the model uses the current word to predict the surrounding window of context words. The skip-gram architecture weighs nearby context words more heavily than more distant context words. The output probabilities are going to relate to how likely it is to find each vocabulary word near our input word. For example, if you gave the trained network the input word “Europe”, the output probabilities are going to be much higher for words like “Belgium” and “Continent” than for unrelated words like “fruits” and “cats”.
# 
# ## How we will use it
# 
# We will first run all the words in each annual report through the word2vec neural network in order to extract a matrix of word embeddings, where each word is theoretically close to semantically related words. We then take a subset of these word embeddings of only words that belong in a given company filing, and average them. This produces a pseudo-document vector which theoretically represents these companies semantically.
# 
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

# Hide warnings
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("../data/preprocessed.csv")


# Here we split each document converted into word vectors.

# In[2]:


from gensim.models.word2vec import Word2Vec
from gensim import utils

bd_processed = df["coDescription_stopwords"].apply(lambda x: utils.simple_preprocess(x))
bd_processed.head()


# Now lets build the Word2Vec model! Due to the sheer amount of computation required, we will limit wach word vector produced to just 200 dimensions. Studies have shown that increasing this size beyond 200 - 300 does not bring much measurable benefit.

# In[3]:


model_w = Word2Vec(bd_processed, vector_size=200)


# We can examine words and see which words are most similar. Below are the most similar words to `cloud`, `trial`, and `oil`.

# In[4]:


model_w.wv.most_similar(positive =['cloud'], topn = 5)


# In[5]:


model_w.wv.most_similar(positive =['trial'], topn = 5)


# In[6]:


model_w.wv.most_similar(positive =['oil'], topn = 5)


# Now we'll map these word vectors back to each document, by averaging all the word vectors that belong to words in a given document (filing)

# In[7]:


def doc_to_vec(text):
    word_vecs = [model_w.wv[w] for w in text if w in model_w.wv]
    
    if len(word_vecs) == 0:
        return np.zeros(model_w.vector_size)
    
    return np.mean(word_vecs, axis = 0)

doc_vec = pd.DataFrame(bd_processed.apply(doc_to_vec).tolist())
labels = np.asarray(model_w.wv.index_to_key)


# If you're interested, the entire 200 dimensions of each document is below:

# In[8]:


doc_vec


# ## Plotting the results
# 
# Here are the results of the word2vec semantic company embedding after dimensionality reduction using PCA.

# In[9]:


plot_pca = std_func.pca_visualize_2d(doc_vec, df.loc[:,["name","SIC_desc"]])


# In[10]:


std_func.pca_visualize_3d(plot_pca)


# conf_mat = std_func.conf_mat(tfidf,df)conf_mat = std_func.conf_mat(tfidf,df)As you can see, these company embeddings don't like quite that great in our reduced space. Perhaps they didn't capture the semantic meaning very well, or its accurate and the semantic embedding of many companies is very jumbled up and their industry classification isn't entirely correct.

# ##  Performance Evaluation 

# In[11]:


conf_mat = std_func.conf_mat(doc_vec,df)


# In[12]:


dot_product_df, accuracy, cm = std_func.dot_product(doc_vec,df)
from sklearn.metrics import classification_report
print(classification_report(dot_product_df["y_true"], dot_product_df["y_pred"], target_names=df["SIC_desc"].unique()))


# From the confusion matrix and the classification report, we can conclude that the word2vec pseudo-company embedding does a poor job at classifying the category of the companies, except for the Crude Petroleum. This is in line with our observations of the PCA plots, as they did not do a very good job at separating companies in different industries.
