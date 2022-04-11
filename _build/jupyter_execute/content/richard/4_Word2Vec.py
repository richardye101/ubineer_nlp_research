#!/usr/bin/env python
# coding: utf-8

# # Word2vec
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


# Here we split each document by word to create a word vector

# In[2]:


from gensim.models.word2vec import Word2Vec
from gensim import utils

revs_processed = df["coDescription_stopwords"].apply(lambda x: utils.simple_preprocess(x))
revs_processed.head()


# Now lets build the Word2Vec model

# In[3]:


model_w = Word2Vec(revs_processed, vector_size=200)


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

doc_vec = pd.DataFrame(revs_processed.apply(doc_to_vec).tolist())
labels = np.asarray(model_w.wv.index_to_key)


# If you're interested, the entire 200 dimensions of each document is below:

# In[8]:


doc_vec


# In[9]:


plot_pca = std_func.pca_visualize_2d(doc_vec, df.loc[:,["name","SIC_desc"]])


# In[10]:


std_func.pca_visualize_3d(plot_pca)


# In[11]:


from gensim.models import doc2vec
from collections import namedtuple

# Load data

# doc1 = ["This is a sentence", "This is another sentence"]

# Transform data (you can add more data preprocessing steps) 

docs = []
analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
for i, text in enumerate(df["coDescription"]):
    words = text.lower().split()
    tags = [i]
    docs.append(analyzedDocument(words, tags))

# Train model (set min_count = 1, if you want the model to work with the provided example data set)

model = doc2vec.Doc2Vec(docs, vector_size = 100, window = 10, min_count = 1, workers = 4)


# In[12]:


# Get the vectors

doc_vec_2 = pd.DataFrame([model.dv[doc] for doc in np.arange(0,len(docs))])
doc_vec_2


# In[13]:


plot_pca_doc2vec = visualize_pca(doc_vec_2, df.loc[:,["name","SIC_desc"]])


# In[14]:


fig = px.scatter_3d(plot_pca_doc2vec[1], x =0 , y = 1, z = 2, hover_data={"name": plot_pca_doc2vec[1].index.get_level_values(0),
                                                              "industry": plot_pca_doc2vec[1].index.get_level_values(1)},
                    color = plot_pca_doc2vec[1].index.get_level_values(1), width=1200, height=700)
fig.show()


# # confusion matrix/ accuracy measure?
# perhaps use KNN and comapre to the cosine similarity work
# 
# collect all the work to get a good big picture idea of our progress
