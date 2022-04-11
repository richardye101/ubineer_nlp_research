#!/usr/bin/env python
# coding: utf-8

# # Embedding methods to augment with dynamic analysis
# 
# This notebook aims to look at the word embeddings for companies and look at whats changed between the years 2016 to 2018 in terms of word.
# <!-- - tf-idf (term frequency - inverse document frequency)
# - LDA (Latent Dirichlet Allocation)
# - LSA (Latent Semantic Analysis)
# - word2vec
# - doc2vec -->

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

data = pd.read_csv("../data/preprocessed.csv")
data_t = pd.read_csv("../data/timeseries_data_2.csv")


# Here are the five companies we wish to look at as a proof of concept: `OKTA`, `Z-Scaler`, `NFLX`, `IBM` and `GE`

# In[2]:


the_five = pd.DataFrame.from_dict(
    {1660134:"OKTA",1713683:"Z-Scaler",1065280:"NFLX",51143:"IBM",40545:"GE"},
    orient = "index").reset_index().rename(columns={"index":"CIK",0:"name"})

data_five_raw = pd.read_csv("../data/bq_dynamic_five.csv")
data_five_raw["CIK"] = data_five_raw["financialEntity"].str.split("=", expand = True).iloc[:,1]
data_five = std_func.clean(data_five_raw)
data_five.head()


# In[3]:


# The companies with names have them, the ones that don't are NaN
clean = pd.merge(
    pd.merge(data_t,data.loc[:,["CIK","name"]], how = "left", on = "CIK"),
    the_five, how = "left", on="CIK")

clean['name'] = clean['name_y'].fillna(clean['name_x'])
clean = clean.drop(["name_x","name_y","reports"],axis = 1)
clean.head()


# In[4]:


final = pd.concat([clean,data_five], axis = 0)
final = final.merge(pd.Series(final.groupby("CIK").size(), name = "reports"), how = "left", on = "CIK").sort_values(["CIK","filingDate"]).reset_index()
final.head()


# ## Diving into the embeddings
# 
# ### tf-idf (term frequency - inverse document frequency)

# In[5]:


import functools
import operator
from datetime import datetime

def deltas(final, embedding, features):
    ignore_words = ["revenue","fiscal","year", "operating", "december", "ended", "administrative", "month", "company", "general", "also",
                    "statement", "asset", "result", "term", "september", "accounting", "million"]
    changes = [[],[],[],[]]
    for i in final.loc[:,"CIK"]:
        # i = final.loc[2,"CIK"]
        # Get the all company filings
        company_filings = embedding[embedding["CIK"] == i].reset_index(drop=True)
        # Get the change YoY in tfidf values
        delta = pd.DataFrame(np.array(company_filings.iloc[1:,3:]) - np.array(company_filings.iloc[:-1,3:]), columns=features)
        # named_delta = pd.concat([company_filings.loc[1:,["filingDate","CIK", "name"]].reset_index(drop=True),delta], axis = 1)
        # Get the top 20 changed terms in YoY filings
        for j in np.arange(company_filings.shape[0]-1):
            word_delta = delta.iloc[j,:].sort_values(key=abs, ascending = False).reset_index()
            word_delta['flagCol'] = np.where(word_delta.loc[:,"index"].str.contains('|'.join(ignore_words)),1,0)
            words = word_delta[word_delta['flagCol'] == 0].iloc[:,:2].head(20).reset_index(drop=True).rename(columns = {"index":"topic",0:"delta"})
            # year = datetime.strptime(company_filings.loc[j,"filingDate"], '%Y-%m-%d %H:%M:%S UTC').date().year
            info = pd.concat([pd.Series(i).repeat(20),pd.Series(str("year " + j + " to year " + int(j+1))).repeat(20)], axis = 1)                 .reset_index(drop=True)                .rename(columns = {0:"CIK",1:"years"})
            to_append = pd.concat([info,words], axis = 1)
            for k in np.arange(to_append.shape[1]):
                changes[k].append(to_append.iloc[:,k].tolist())

    for i in np.arange(len(changes)):
        changes[i] = functools.reduce(operator.iconcat, changes[i], [])
        
    return(pd.DataFrame(list(zip(changes[0],changes[1],changes[2],changes[3]))))


# In[6]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

# combine the techniques since tf-idf only augments count vectorized documents
pipe = Pipeline([('count', CountVectorizer(ngram_range = (2,4), max_features = 1000)),
                  ('tfidf', TfidfTransformer())]).fit(final["coDescription_stopwords"])

tfidf = pd.DataFrame(pipe.transform(final["coDescription_stopwords"]).toarray())
data_tfidf = pd.concat([final.loc[:,["filingDate","CIK", "name"]],tfidf], axis = 1)
delta_tfidf = deltas(final, data_tfidf, pipe.get_feature_names_out().tolist())
delta_tfidf


# In[7]:


the_five


# In[8]:


# OKTA
delta_tfidf[delta_tfidf.iloc[:,0] == 1660134].iloc[:60,:].groupby(1).head(5)


# In[9]:


# Z-Scaler
delta_tfidf[delta_tfidf.iloc[:,0] == 1713683].iloc[:60,:].groupby(1).head(5)


# In[10]:


final[final["CIK"] == 1065280]


# In[11]:


# Netflix
delta_tfidf[delta_tfidf.iloc[:,0] == 1065280].iloc[:60,:].groupby(1).head(5)


# In[12]:


# GE
delta_tfidf[delta_tfidf.iloc[:,0] == 40545].iloc[:60,:].groupby(1).head(5)


# ### word2vec

# In[13]:


get_ipython().system('source /Users/richardye/Documents/Python/venv_ubineer/bin/activate')
# !pip3 install gensim


# In[14]:


from gensim.models.word2vec import Word2Vec
from gensim import utils

revs_processed = final["coDescription_stopwords"].apply(lambda x: utils.simple_preprocess(x))

# https://stackoverflow.com/questions/46560861/relation-between-word2vec-vector-size-and-total-number-of-words-scanned
model_w = Word2Vec(revs_processed, vector_size = 300)

def doc_to_vec(text):
    word_vecs = [model_w.wv[w] for w in text if w in model_w.wv]
    
    if len(word_vecs) == 0:
        return np.zeros(model_w.vector_size)
    
    return np.mean(word_vecs, axis = 0)

doc_vec = pd.DataFrame(revs_processed.apply(doc_to_vec).tolist())


# In[15]:


doc_vec


# In[16]:


first_yr = pd.concat([final.loc[:,["filingDate","CIK", "name"]],doc_vec],axis = 1).groupby("CIK").head(1)
print(first_yr.shape)
first_yr.head(5)


# In[17]:


from sklearn.cluster import DBSCAN

clustering = DBSCAN(eps=.05, min_samples=5, metric='cosine').fit(first_yr.iloc[:,3:])
# clustering.labels_


# In[18]:


clustering.labels_.max()


# In[19]:


unique_elements, counts_elements = np.unique(clustering.labels_, return_counts=True)
print("Frequency of unique values of the said array:")
print(np.asarray((unique_elements, counts_elements)))


# In[20]:


centers_w2v = pd.concat([first_yr.iloc[:,3:],pd.Series(clustering.labels_, name = "cluster")], axis = 1)     .groupby("cluster").mean()
centers_w2v.head()


# In[21]:


from sklearn.metrics.pairwise import cosine_similarity
# get distances to each cluster center created based on first year filings
# want to find representative companies for each cluster
cosine_dist = pd.concat([first_yr.loc[:,["filingDate","CIK", "name"]].reset_index(drop=True),
                         pd.DataFrame(cosine_similarity(first_yr.iloc[:,3:],centers_w2v))],axis = 1)
cosine_dist
# pd.concat([first_yr.loc[:,["filingDate","CIK", "name"]],
#            pd.DataFrame(cosine_similarity(first_yr.iloc[:,3:],centers_w2v))], axis = 1)


# In[22]:


cluster_companies = []
for i in np.arange(3,cosine_dist.shape[1]):
    rank = cosine_dist.iloc[:,i].sort_values(ascending = False).index.tolist()[:5]
    CIK = first_yr.iloc[rank,1]
    rep_words = final[final["CIK"].isin(CIK)].groupby("CIK").head(1).loc[:,"coDescription"]
    cluster_companies.append(rep_words)
cluster_companies[:3]


# In[23]:


all_w2v = pd.concat([final.loc[:,["filingDate","CIK", "name"]],doc_vec],axis = 1)
yr_2_dist = all_w2v[all_w2v.groupby("CIK").cumcount() == 1]
yr_3_dist = all_w2v[all_w2v.groupby("CIK").cumcount() == 2]

cosine_dist_2 = pd.concat([yr_2_dist.loc[:,["filingDate","CIK", "name"]].reset_index(drop=True),
                           pd.DataFrame(cosine_similarity(yr_2_dist.iloc[:,3:],centers_w2v))],axis = 1)
cosine_dist_3 = pd.concat([yr_3_dist.loc[:,["filingDate","CIK", "name"]].reset_index(drop=True),
                           pd.DataFrame(cosine_similarity(yr_3_dist.iloc[:,3:],centers_w2v))],axis = 1)


# In[24]:


y1_y2 = pd.DataFrame(np.array(cosine_dist_2[cosine_dist_2.loc[:,"CIK"].isin(cosine_dist.loc[:,"CIK"])].iloc[:,3:]) -     np.array(cosine_dist[cosine_dist.loc[:,"CIK"].isin(cosine_dist_2.loc[:,"CIK"])].iloc[:,3:]),
                     index = pd.MultiIndex.from_frame(pd.DataFrame(cosine_dist.loc[cosine_dist.loc[:,"CIK"].isin(cosine_dist_2.loc[:,"CIK"]), "CIK"]), names=["CIK"])).reset_index()
y1_y2


# In[25]:


the_five


# In[26]:


y1_y2[y1_y2["CIK"].isin(the_five["CIK"])].sort_values(by=36, axis =1, key = abs, ascending = False)


# In[27]:


[w[:300] for w in cluster_companies[2].tolist()]


# In[28]:


y2_y3 = pd.DataFrame(np.array(cosine_dist_3[cosine_dist_3.loc[:,"CIK"].isin(cosine_dist_2.loc[:,"CIK"])].iloc[:,3:]) -     np.array(cosine_dist_2[cosine_dist_2.loc[:,"CIK"].isin(cosine_dist_3.loc[:,"CIK"])].iloc[:,3:]),
                     index = pd.MultiIndex.from_frame(pd.DataFrame(cosine_dist_2.loc[cosine_dist_2.loc[:,"CIK"].isin(cosine_dist_3.loc[:,"CIK"]), "CIK"]), names=["CIK"])).reset_index()
y2_y3


# In[29]:


y2_y3[y2_y3["CIK"].isin(the_five["CIK"])].sort_values(by=32, axis =1, key = abs, ascending = False)


# In[30]:


std_func.pca_visualize_2d(doc_vec, pd.DataFrame(clustering.labels_))


# In[31]:


model_w.wv.most_similar(positive =['ibm'])


# The numer of features is actually the number of dimensions, so there are 300 "topics" in this space currently. We'll reduce that before moving forward.

# In[32]:


# Since its not sparse, PCA should work just fine
multi_index = pd.MultiIndex.from_frame(final.loc[:,["filingDate","CIK", "name"]])
    
from sklearn.decomposition import PCA
pca = PCA(n_components = 10)
pca_embedding = pca.fit_transform(doc_vec)
pca_embedding = pd.DataFrame(pca_embedding, index = multi_index).reset_index()
pca_embedding


# In[33]:


data_word2vec = pd.concat([final.loc[:,["filingDate","CIK", "name"]],doc_vec], axis = 1)
delta_word2vec = deltas(final, data_word2vec, model_w.wv.index_to_key)
delta_word2vec


# In[34]:


data_word2vec


# ### doc2vec

# In[35]:


from gensim.models import doc2vec
from collections import namedtuple

docs = []
analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
for i, text in enumerate(final["coDescription_stopwords"]):
    words = text.lower().split()
    tags = [i]
    docs.append(analyzedDocument(words, tags))

# Train model (set min_count = 1, if you want the model to work with the provided example data set)

model = doc2vec.Doc2Vec(docs, vector_size = 100, window = 10, min_count = 1, workers = 4)


# In[36]:


doc_vec_2 = pd.DataFrame([model.dv[doc] for doc in np.arange(0,len(docs))])
doc_vec_2


# In[37]:


import hdbscan
clusterer = hdbscan.HDBSCAN()
clusterer.fit(doc_vec_2)


# In[38]:


clusterer.labels_.max()

