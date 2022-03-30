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
from sklearn.decomposition import TruncatedSVD
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

def visualize_svd(vectors, index):
    multi_index = pd.MultiIndex.from_frame(index, names=["name", "industry"])
    
    svd = TruncatedSVD(n_components = min(10,vectors.shape[1]))
    svd_embedding = svd.fit_transform(vectors)
    svd_embedding = pd.DataFrame(svd_embedding, index = multi_index)
    
    fig = px.scatter(svd_embedding, x =0 , y = 1, hover_data={"name": svd_embedding.index.get_level_values(0),
                                                              "industry": svd_embedding.index.get_level_values(1)},
                     color = svd_embedding.index.get_level_values(1), width=1200, height=600)
    fig.show()
    
    return [svd, svd_embedding]

def pca_visualize_3d(plot):
    if(plot[1].index.nlevels == 1):
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


## Accuracy measures
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
def dot_product(embedding_matrix, data):
    """calculates percentage of correct category predictions based on 1-NN using dot product
    
    args: embedding matrix of size nxm (n companies each with an embedding of size m). NOTE: embeddings should be normalized.
    
    returns: float representation of percentage of correct category predictions
    """
    dot_product = np.matmul(embedding_matrix, embedding_matrix.T)
    np.fill_diagonal(dot_product.values, 0)
    dot_product.index = data["SIC_desc"]
    dot_product.columns = data["SIC_desc"]
    dot_product_df = pd.DataFrame(dot_product.idxmax(axis=1))
    dot_product_df.reset_index(level=0, inplace=True)
    dot_product_df.columns = ["y_true", "y_pred"]
    return dot_product_df, np.sum(np.where(dot_product_df.iloc[:,1] == dot_product_df.iloc[:,0], 1, 0))/len(embedding_matrix), confusion_matrix(dot_product_df["y_true"], dot_product_df["y_pred"], labels=None, sample_weight=None, normalize='true')

def conf_mat(matrix, data):
    dot_product_df, accuracy, cm = dot_product(matrix, data)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=data["SIC_desc"].unique())
    disp.plot(xticks_rotation='vertical')
    
    return dot_product_df
    
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

def show_ROC_curves(df, similarity_matrix):
    for i in df["SIC_desc"].unique():
        y_true = similarity_matrix["y_true"] == i
        y_pred = similarity_matrix["y_pred"] == i
        fpr,tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc),
        )
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Some extension of Receiver operating characteristic to multiclass")
    plt.legend(loc="lower right")
    plt.show()

def get_topics(model, vectorizer, num_topics):
    
    #the word ids obtained need to be reverse-mapped to the words so we can print the topic names.
    feat_names = vectorizer.get_feature_names()
    
    word_dict = {};
    for i in range(num_topics):
        
        #for each topic, obtain the largest values, and add the words they map to into the dictionary.
        words_ids = model.components_[i].argsort()[:-20 - 1:-1]
        words = [feat_names[key] for key in words_ids]
        word_dict['Topic # ' + '{:02d}'.format(i+1)] = words;
    
    return pd.DataFrame(word_dict);