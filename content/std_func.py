import os
import json
import pandas as pd
import numpy as np
import re
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('omw-1.4', quiet=True)
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
import plotly.express as px
import pattern
import collections
from pattern.en import parsetree, singularize

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
                         color = pca_embedding.index.get_level_values(1), width=900, height=700)
    else:
        fig = px.scatter(pca_embedding, x =0 , y = 1, hover_data={"name": pca_embedding.index.get_level_values(0)},
                         color = pca_embedding.index, width=900, height=700)
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y = -.25,
        xanchor="center",
        x = .5
    ))
    fig.show()
    
    return [pca, pca_embedding]

def visualize_svd(vectors, index):
    multi_index = pd.MultiIndex.from_frame(index, names=["name", "industry"])
    
    svd = TruncatedSVD(n_components = min(10,vectors.shape[1]))
    svd_embedding = svd.fit_transform(vectors)
    svd_embedding = pd.DataFrame(svd_embedding, index = multi_index)
    
    fig = px.scatter(svd_embedding, x =0 , y = 1, hover_data={"name": svd_embedding.index.get_level_values(0),
                                                              "industry": svd_embedding.index.get_level_values(1)},
                     color = svd_embedding.index.get_level_values(1), width=900, height=700)
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y = -.25,
        xanchor="center",
        x = .5
    ))
    fig.show()
    
    return [svd, svd_embedding]

def pca_visualize_3d(plot):
    if(plot[1].index.nlevels == 1):
        fig = px.scatter_3d(plot[1], x =0 , y = 1, z = 2, hover_data={"name": plot[1].index},
                    color = plot[1].index, width=1200, height=700)
    else:
        fig = px.scatter_3d(plot[1], x =0 , y = 1, z = 2, hover_data={"name": plot[1].index.get_level_values(0),
                                                              "industry": plot[1].index.get_level_values(1)},
                    color = plot[1].index.get_level_values(1), width=900, height=700)
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y = -.25,
        xanchor="center",
        x = .5
    ))
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

## POS-tagging
def extract_nouns(t):
    tree = parsetree(t)
    nouns = []
    for sentence in tree:
        for word in sentence:
            if 'NN' in word.type:
                nouns.append(singularize(word.string))
    return " ".join(nouns)

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
    
def get_accuracy(cosine_matrix, data):
    np.fill_diagonal(cosine_matrix.values, 0)
    cosine_matrix.index = data["SIC_desc"]
    cosine_matrix.columns = data["SIC_desc"]
    prediction = pd.DataFrame(cosine_matrix.idxmax(axis=1))
    prediction.reset_index(level=0, inplace=True)
    prediction.columns = ["y_true","y_pred"]
    return (prediction, np.sum(np.where(prediction.iloc[:,1] == prediction.iloc[:,0], 1, 0))/len(prediction),
            confusion_matrix(prediction["y_true"], prediction["y_pred"], labels=None, sample_weight=None, normalize='true'))

def conf_mat_cosine(matrix, data):
    prediction, accuracy, cm = get_accuracy(matrix, data)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=data["SIC_desc"].unique())
    disp.plot(xticks_rotation='vertical')

    return prediction


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
        words_ids = model.components_[i].argsort()[:-10 - 1:-1]
        words = [feat_names[key] for key in words_ids]
        word_dict['Topic # ' + '{:02d}'.format(i+1)] = words;
    
    return pd.DataFrame(word_dict);

def graph_netflix(num_topics, netflix_top_df):
    plt.rcParams['figure.figsize'] = [12, 7]
    X_axis = np.arange(num_topics)
    plt.bar(X_axis - 0.5,list (netflix_top_df.iloc[0]), 0.15, label="2006", color="lightcyan")
    plt.bar(X_axis - 0.3, list(netflix_top_df.iloc[1]), 0.15, label="2007", color="paleturquoise")
    plt.bar(X_axis - 0.1, list(netflix_top_df.iloc[2]), 0.15, label="2008", color="mediumturquoise")
    plt.bar(X_axis + 0.1, list(netflix_top_df.iloc[3]), 0.15, label="2009", color="teal")
    plt.bar(X_axis + 0.3, list(netflix_top_df.iloc[4]), 0.15, label="2010", color="darkslategrey")
    plt.bar(X_axis + 0.5, list(netflix_top_df.iloc[5]), 0.15, label="2011", color="black")
    plt.xticks(X_axis, [str(i) for i in range(1,num_topics+1)])
    plt.xlabel("Topics")
    plt.ylabel("Weight")
    plt.legend()
    plt.show()
    
def graph_ge(num_topics, ge_top_df):
    plt.rcParams['figure.figsize'] = [12, 7]
    X_axis = np.arange(num_topics)
    plt.bar(X_axis - 0.3, list(ge_top_df.iloc[0]), 0.2, label="2011", color="paleturquoise")
    plt.bar(X_axis-0.1, list(ge_top_df.iloc[1]), 0.2, label="2012", color="mediumturquoise")
    plt.bar(X_axis + 0.1, list(ge_top_df.iloc[2]), 0.2, label="2013", color="teal")
    plt.bar(X_axis + 0.3, list(ge_top_df.iloc[3]), 0.2, label="2014", color="darkslategrey")
    plt.xticks(X_axis, [str(i) for i in range(1,num_topics+1)])
    plt.xlabel("Topics")
    plt.ylabel("Weight")
    plt.legend()
    plt.show()

def get_differences(topics, series1, series2):
    df = pd.DataFrame(series1 - series2).reset_index(drop=True)
    df.columns = ["weight_diff"]
    df = df.join(topics.T.reset_index(drop=True))
    df.index = ["Topic #" + str(i) for i in range(1,topics.shape[1]+1)]
    df["words"] = df.iloc[:,1:11].apply(lambda x: ', '.join(x), axis=1)
    df = df[["weight_diff","words"]]
    df.sort_values(by=["weight_diff"], key=abs, ascending=False, inplace=True)
    return df  

## dynamic company embedding code

import functools
import operator
from datetime import datetime

def deltas(final, embedding, features):
    ignore_words = ["revenue","fiscal","year", "operating", "december", "ended", "administrative", "month", "company", "general", "also",
                    "statement", "asset", "result", "term", "september", "accounting", "million"]
    changes = [[],[],[],[],[]]
    for i in final.loc[:,"CIK"]:
        # i = final.loc[2,"CIK"]
        company_name = final[final["CIK"] == i].loc[:,"Name"].unique()[0]
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
            info = pd.concat([pd.Series(i).repeat(20), pd.Series(company_name).repeat(20),
                              pd.Series(str("year " + str(j) + " to year " + str(j+1))).repeat(20)], axis = 1) \
                .reset_index(drop=True)\
                .rename(columns = {0:"CIK",1:"years"})
            to_append = pd.concat([info,words], axis = 1)
            for k in np.arange(to_append.shape[1]):
                changes[k].append(to_append.iloc[:,k].tolist())

    for i in np.arange(len(changes)):
        changes[i] = functools.reduce(operator.iconcat, changes[i], [])
        
    return(pd.DataFrame(list(zip(changes[0],changes[1],changes[2],changes[3], changes[4]))))

def dynamic_plt(company_cosine, industries, company):
    fig = px.line(pd.melt(company_cosine.loc[:,["filingDate"] + list(industries)],
                               id_vars = "filingDate", value_vars = list(industries)),
                       x = "filingDate",
                       y = "value",
                       color = "variable",
                       title = company + "'s " + "distance to it's closest industries over time",
                       width=900,
                       height=700,
                       markers = True)
    fig.update_layout(legend=dict(
        title = "Industry",
        orientation="h",
        yanchor="bottom",
        y = -.4,
        xanchor="center",
        x = .5),
        yaxis_title = "Cosine Similarity",)
    fig.show()