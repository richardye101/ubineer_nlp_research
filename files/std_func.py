import os
import json
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
# nltk.download('stopwords')
# nltk.download('wordnet')
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px

# Referenced from my CSCD25 course
def pca_visualize_2d(vectors, index):
    if(index.shape[1] ==2):
        multi_index = pd.MultiIndex.from_frame(index, names=["name", "industry"])
    else:
        multi_index = pd.Index(index.iloc[:,0])
    pca = PCA(n_components = min(10,vectors.shape[1]))
    pca_embedding = pca.fit_transform(vectors)
    pca_embedding = pd.DataFrame(pca_embedding, index = multi_index)
    
    if(index.shape[1] ==2):
        fig = px.scatter(pca_embedding, x =0 , y = 1, hover_data={"name": pca_embedding.index.get_level_values(0),
                                                                  "industry": pca_embedding.index.get_level_values(1)},
                         color = pca_embedding.index.get_level_values(1), width=1200, height=600)
    else:
        fig = px.scatter(pca_embedding, x =0 , y = 1, hover_data={"name": pca_embedding.index.get_level_values(0)},
                         color = pca_embedding.index, width=1200, height=600)
    fig.show()
    
    return [pca, pca_embedding]

def pca_visualize_3d(plot):
    if(plot[1].index.ndim == 1):
        fig = px.scatter_3d(plot[1], x =0 , y = 1, z = 2, hover_data={"name": plot[1].index},
                    color = plot[1].index, width=1200, height=600)
    else:
        fig = px.scatter_3d(plot[1], x =0 , y = 1, z = 2, hover_data={"name": plot[1].index.get_level_values(0),
                                                              "industry": plot[1].index.get_level_values(1)},
                    color = plot[1].index.get_level_values(1), width=1200, height=600)
    fig.show()

#strip any left over html code
def clean_data_fn(insrt_data):
    clean_data = []
    for idx, ele in insrt_data.iterrows():
        if "https://www.sec.gov/Archives/edgar/data/" in ele["coDescription"]:
            pass
        else:
            clean_txt = re.compile('<.*?>')
            desc = re.sub(clean_txt,'',ele["coDescription"]).replace(u'\xa0', u' ').replace("   ", "").replace("'", "").replace('"','')
            if re.search('<', desc):
                pos = re.search('<', desc).start()
            desc = desc[:pos].lower()
            if (desc.find("business") >= 20): # didnt find it in the first 20 characters then look for next
                desc = desc[6 : ( desc.rfind("<") )] # remove the "Item 1." stuff only
            else: # found "business", remove everything before it
                desc =  desc[( desc.find("business") + 8 ) : ( desc.rfind("<") ) ]
            if (desc.find("overview") <= 20): # didnt find it in the first 20 characters then look for next
                desc =  desc[( desc.find("overview") + 8 ) :]
            # remove leading white space and periods
            desc = re.sub(r"^\.", "", desc).strip()            
            new_data = ele.copy()
            new_data["coDescription"] = desc
            # remove any filings with a description less than 250 characters (not enough information for us)
            if len(desc)<250:
                pass
            else:
                clean_data.append(new_data)
                
    return(pd.DataFrame(clean_data))

def lemmatize_sentence(sentence):
    #lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_output = [lemmatizer.lemmatize(w) for w in word_tokenize(sentence)]
    return " ".join(lemmatized_output)

# remove all numbers so they don't show up as dimensions
def remove_nums(x):
    text = x.lower()
    text = re.sub(r'\d+', '', text)
    return text

# remove stopwords and punctuation
def remove_stopwords(x):
    stop_words = set(stopwords.words('english'))

    word_tokens = word_tokenize(x)

    filtered_sentence = ' '.join([w for w in word_tokens if not w.lower() in stop_words and w.isalnum()])

    return(filtered_sentence)

def clean(df):
    non_html_data = clean_data_fn(df)#.rename(columns = {"financialEntity":"CIK"})
    non_html_data["CIK"] = non_html_data["CIK"].astype(int)
    
    lemma_desc = non_html_data["coDescription"].apply(lemmatize_sentence)
    non_html_data["coDescription_lemmatized"] = lemma_desc
    non_html_data["coDescription_lemmatized"].head()

    rm_num_stopwords = non_html_data["coDescription_lemmatized"].apply(remove_nums).apply(remove_stopwords)
    non_html_data["coDescription_stopwords"] = rm_num_stopwords
    
    return non_html_data

