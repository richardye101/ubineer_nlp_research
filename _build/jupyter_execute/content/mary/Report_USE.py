#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.insert(0, '..')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('aimport', 'std_func')
import pandas as pd
import numpy as np
import plotly.express as px

df = pd.read_csv("../data/preprocessed.csv")
df.drop_duplicates(subset = "name", keep=False, inplace=True)


# # Universal Sentence Encoder
# The Universal Sentence Encoder encodes text into high-dimensional vectors that can be used for text classification, semantic similarity, clustering and other natural language tasks. It is a pre-trained model created by Google that uses sources like Wikipedia, web news, web question-answer pages, and discussion forums. There are two variations of the model, one trained with Transformer encoder and the other with Deep Averaging Network (DAN). The one with Transformer encoder is computationally more intensive but provides better results, while DAN trades accuracy for lower computational requirements. In our works, the model with DAN has provided results with high accuracy so we do not require the Transformer encoder alternative. The input is a variable-length English text and the output is a normalised 512-dimensional vector.

# In[2]:


embeddings = pd.read_csv('embeddings.csv', index_col=0)
embeddings.head()


# ## Plotting

# In[3]:


plot_d2v = std_func.pca_visualize_2d(embeddings, df.loc[:,["name","SIC_desc"]])
std_func.pca_visualize_3d(plot_d2v)


# ##  Performance Evaluation 
# USE gives normalized embeddings, so the inner product of encodings can be treated as a similarity matrix.

# In[4]:


dot_product_df, accuracy, cm = std_func.dot_product(embeddings,df)


# In[5]:


from sklearn.metrics import classification_report
print(classification_report(dot_product_df["y_true"], dot_product_df["y_pred"], target_names=df["SIC_desc"].unique()))


# In[6]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=df["SIC_desc"].unique())
disp.plot(xticks_rotation=90)

