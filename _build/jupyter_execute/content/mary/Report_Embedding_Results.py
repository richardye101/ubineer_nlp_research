#!/usr/bin/env python
# coding: utf-8

# ## Document Embedding Results 
# 
# The table below illustrates an overview of the results of all the techniques we explored to create document embeddings. 

# <!-- |Embedding Technique | Prepackaged Software(Recall) |  Crude Petroleum and Natural Gas(Recall) |  Pharmaceutical Preparations(Recall) | Real Estate Investment Trusts(Recall) | State Commercial Banks(Recall) | Weighted Average(Recall) 
# |--------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|
# |TF-IDF| 0.89 | 0.94 | 0.75 | 0.94 | 0.95 | 0.91 | 0.89 | 0.91 |
# |N-grams - Cosine Similarity|  0.85 | 0.93  |  0.76 |  0.95 |  0.94 |  0.91 |
# |POS Tagging - Cosine Similarity| 0.86  |  0.96 |  0.91 |  0.97 | 0.97  | 0.95  |
# |Word2Vec| 0.76 | 0.95 | 0.46 | 0.85 | 0.82 | 0.82 | 0.77 | 0.82 |
# |Doc2Vec| 0.88 | 0.97 | 0.86 | 0.91 | 0.94 | 0.92 | 0.91 | 0.92 |
# |TwoTowers| 0.67  | 0.56  |0.60   |0.68   |0.65   |0.63   |
# |Universal Sentence Encoder| 0.83  |0.96  |0.90   |0.96   |0.96 |0.94 | -->

# In[1]:


cols = ["Prepackaged Software(Recall)", "Crude Petroleum and Natural Gas(Recall)",
        "Pharmaceutical Preparations(Recall)", "Real Estate Investment Trusts(Recall)", "State Commercial Banks(Recall)",
        "Weighted Average(Recall)"]
data = [[0.89,0.94,0.75,0.94,0.95,0.91],
[0.85,0.93,0.76,0.95,0.94,0.91],
[0.86,0.96,0.91,0.97,0.97,0.95],
[0.76,0.95,0.46,0.85,0.82,0.82],
[0.88,0.97,0.86,0.91,0.94,0.92],
[0.67,0.56,0.60,0.68,0.65,0.63],
[0.83,0.96,0.90,0.96,0.96,0.94]]
# [[str(i) for i in row] for row in data]
methods = ["TF-IDF", "N-grams - Cosine Similarity", "POS Tagging - Cosine Similarity", "Word2Vec",
           "Doc2Vec","TwoTowers", "Universal Sentence Encoder"]

import pandas as pd
df = pd.DataFrame(data, index = methods, columns = cols).T.round(2)     .rename_axis("Recall/Sensitivity of Industry").rename_axis("Embedding Technique", axis = "columns")

import seaborn as sns
cm = sns.light_palette("#5CCDC6", n_colors = 35, as_cmap=True)

df.style.background_gradient(cmap=cm)


# ## Conclusion
# To be added when we have all results.
# probably conclude that most of our model done well and is good at classifying xxx category.
