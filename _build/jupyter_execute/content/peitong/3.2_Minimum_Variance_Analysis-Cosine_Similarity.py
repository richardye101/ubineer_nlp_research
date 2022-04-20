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


# ## Estimates from Cosine Similarity
# We want to evaluate the feasibility of constructing optimized portfolios with the word embedding results. Our first estimate on the textual analysis is generating optimal portfolios using cosine similarity distances. We use the cosine similarity distance as correlation and sample return standard deviation to calculate the covariance estimate. We will compare the results at the end of this section to determine the feasibility.

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


r_selected = pd.read_csv("data/filtered_r.csv")
# get the mean of all 
r_selected.set_index("name", inplace = True)
mu = r_selected.mean(axis = 1)
# compute the covariance matrix 
cov = r_selected.T.cov()


# ### Cosine Similarity Distances
# We conduct cosine similarity analysis with 2-to-4 grams embeddings on the business description of each company for all top 5 SIC industry. First, we generate the words counting matrix and perform cosine similarity anlaysis to calculate the distances, which will be used as the correlation between companies in the next step for generating covarince estimate. 

# In[4]:


df = pd.read_csv('../data/preprocessed.csv',
                 usecols = ['reportingDate', 'name', 'CIK',
                           'coDescription_stopwords', 'SIC', 'SIC_desc'])
df = df.set_index(df.name)


# #### Words Count
# For this cosine similarity analysis, we generate sequences of 2 to 4 words as one term and only select the top 600 terms by frequency.

# In[5]:


from sklearn.feature_extraction.text import CountVectorizer

Vectorizer = CountVectorizer(ngram_range = (2,4), 
                             max_features = 600)

count_data = Vectorizer.fit_transform(df['coDescription_stopwords'])
wordsCount = pd.DataFrame(count_data.toarray(),columns=Vectorizer.get_feature_names())
wordsCount = wordsCount.set_index(df['name'])
wordsCount


# #### Cosine Similarity Computation

# In[6]:


# Compute Cosine Similarity
from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = pd.DataFrame(cosine_similarity(wordsCount, wordsCount))
cosine_sim = cosine_sim.set_index(df['name'])
cosine_sim.columns = df['name']
cosine_sim


# ### Perform Mean-Variance Analysis
# We only use the Pharmaceutical Preparations industry data to generate portfolio based on Mean-Variance Analysis. We calculate the covariance estimate with cosine similarity distance as correlation and the sample standard deviation of returns. Then we use the sample return and estimated covariance to build efficient frontier.

# In[7]:


get_ipython().system('pip install PyPortfolioOpt')


# In[8]:


from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import objective_functions
from pypfopt import plotting


# In[9]:


# get the names of the companies in the pharmaceutical preparations industry
Pharm = df[df.SIC == 2834]
Pharm_list = Pharm.index


# In[10]:


# get the companies name that match return data and business description data
SET = (set(Pharm_list) & set(r_selected.index))
LIST = [*SET, ]


# #### Sample Mean for the Pharmaceutical Preparations Industry

# In[11]:


mu_Pharm = mu[LIST]


# #### Sample Covariance for the Pharmaceutical Preparations Industry

# In[12]:


tmp = cov[LIST].T
cov_Pharm = tmp[LIST]


# #### Cosine Similarity Distances for the Pharmaceutical Preparations Industry

# In[13]:


tmp = cosine_sim[LIST].drop_duplicates().T
Pharm_cos_sim = tmp[LIST].drop_duplicates()


# #### Covariance for Cosine Similarity

# In[14]:


sd = pd.DataFrame(np.sqrt(np.diag(np.diagonal(cov_Pharm))))
sd = sd.set_index(cov_Pharm.index)
sd.columns = cov_Pharm.index
cos_sim_cov = pd.DataFrame((np.dot(np.dot(sd, Pharm_cos_sim),sd)))


# #### Efficient Frontier - Pharmaceutical Preparations

# In[15]:


ef1 = EfficientFrontier(mu_Pharm, cos_sim_cov, weight_bounds=(0, 0.2))

fig, ax = plt.subplots()
plotting.plot_efficient_frontier(ef1, ax=ax, show_assets=True)

# Find and plot the tangency portfolio
ef2 = EfficientFrontier(mu_Pharm, cos_sim_cov, weight_bounds=(0, 0.2))
# min volatility
ef2.min_volatility()
ret_tangent, std_tangent, _ = ef2.portfolio_performance()
ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Min Volatility")

# Format
ax.set_title("Efficient Frontier - Pharmaceutical Preparations \n Cosine Similarity Estimates")
ax.legend()
plt.tight_layout()
plt.savefig('images/Efficient_Frontier_Cos_Sim.png', dpi=200, bbox_inches='tight')
plt.show()


# ##### Min Volatility Portfolio

# ###### Performance

# In[16]:


ef2.portfolio_performance(verbose=True);


# ###### Weights

# In[17]:


companies = []
weights = []
for company, weight in ef2.clean_weights().items():
    if weight != 0:
        companies.append(company)
        weights.append(weight)
        
dic = {'Company_Name':companies,'Weight':weights}
min_vol = pd.DataFrame(dic)
min_vol = pd.DataFrame(dic)
min_vol.to_csv("data/min_vol_cos_sim_Pharmaceutical_Preparations.csv", index = False)


# In[18]:


pd.read_csv("data/min_vol_cos_sim_Pharmaceutical_Preparations.csv")


# ### Results for the Other 4 Industries

# In[19]:


SIC_list = [7372, 1311, 6798, 6022]
SIC_desc = ['Prepackaged Software (mass reproduction of software)', 'Crude Petroleum and Natural Gas', 
           'Real Estate Investment Trusts', 'State Commercial Banks (commercial banking)']


# #### Prepackaged Software (mass reproduction of software)

# In[20]:


SIC = SIC_list[0]
    
industry_name = SIC_desc[SIC_list.index(SIC)]
    
# get the names of the companies in the other industries
Companies = df[df.SIC == SIC]
Company_list = Companies.index

# get the companies name that match return data and business description data
SET = (set(Company_list) & set(r_selected.index))
LIST = [*SET, ]

mu_sample = mu[LIST]
# get the outliers
outlier = mu_sample[mu_sample>1].index
mu_sample = mu_sample.drop(outlier)
LIST = mu_sample.index

tmp = cov[LIST].T
cov_sample = tmp[LIST]

tmp = cosine_sim[LIST].T
tmp = tmp[~tmp.index.duplicated(keep="first")]
industry_cos_sim = tmp[LIST].T
industry_cos_sim = industry_cos_sim[~industry_cos_sim.index.duplicated(keep="first")]

sd = pd.DataFrame(np.sqrt(np.diag(np.diagonal(cov_sample))))
sd = sd.set_index(cov_sample.index)
sd.columns = cov_sample.index
cos_sim_cov = pd.DataFrame((np.dot(np.dot(sd, industry_cos_sim),sd)))

# perform minimum variance analysis
ef1 = EfficientFrontier(mu_sample, cos_sim_cov, weight_bounds=(0, 0.2))

fig, ax = plt.subplots()
plotting.plot_efficient_frontier(ef1, ax=ax, show_assets=True)

# Find and plot the tangency portfolio
ef2 = EfficientFrontier(mu_sample, cos_sim_cov, weight_bounds=(0, 0.2))
# min volatility
ef2.min_volatility()
ret_tangent, std_tangent, _ = ef2.portfolio_performance()
ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Min Volatility")

# Format
ax.set_title("Efficient Frontier - %s \n Cosine Similarity Estimates" %industry_name)
ax.legend()
plt.tight_layout()
plt.savefig('images/Efficient_Frontier_Cosine_Similarity_Estimates' + str(industry_name) + '.png', dpi=200, bbox_inches='tight')
plt.show()


# ##### Min Volatility Portfolio

# ###### Performance

# In[21]:


ef2.portfolio_performance(verbose=True);


# ###### Weights

# In[22]:


companies = []
weights = []
for company, weight in ef2.clean_weights().items():
    if weight != 0:
        companies.append(company)
        weights.append(weight)
        
dic = {'Company_Name':companies,'Weight':weights}
min_vol = pd.DataFrame(dic)
min_vol.to_csv("data/min_vol_cos_sim_Prepackaged_Software.csv", index = False)


# In[23]:


pd.read_csv("data/min_vol_cos_sim_Prepackaged_Software.csv")


# #### Crude Petroleum and Natural Gas
# When we conduct the same analysis, there is no weight shown. Efficient frontier cannot be found.

# #### Real Estate Investment Trusts

# In[24]:


SIC = SIC_list[2]
    
industry_name = SIC_desc[SIC_list.index(SIC)]
    
# get the names of the companies in the other industries
Companies = df[df.SIC == SIC]
Company_list = Companies.index

# get the companies name that match return data and business description data
SET = (set(Company_list) & set(r_selected.index))
LIST = [*SET, ]

mu_sample = mu[LIST]
# get the outliers
outlier = mu_sample[mu_sample>1].index
mu_sample = mu_sample.drop(outlier)
LIST = mu_sample.index

tmp = cov[LIST].T
cov_sample = tmp[LIST]

tmp = cosine_sim[LIST].T
tmp = tmp[~tmp.index.duplicated(keep="first")]
industry_cos_sim = tmp[LIST].T
industry_cos_sim = industry_cos_sim[~industry_cos_sim.index.duplicated(keep="first")]

sd = pd.DataFrame(np.sqrt(np.diag(np.diagonal(cov_sample))))
sd = sd.set_index(cov_sample.index)
sd.columns = cov_sample.index
cos_sim_cov = pd.DataFrame((np.dot(np.dot(sd, industry_cos_sim),sd)))

# perform minimum variance analysis
ef1 = EfficientFrontier(mu_sample, cos_sim_cov, weight_bounds=(0, 0.2))

fig, ax = plt.subplots()
plotting.plot_efficient_frontier(ef1, ax=ax, show_assets=True)

# Find and plot the tangency portfolio
ef2 = EfficientFrontier(mu_sample, cos_sim_cov, weight_bounds=(0, 0.2))
# min volatility
ef2.min_volatility()
ret_tangent, std_tangent, _ = ef2.portfolio_performance()
ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Min Volatility")

# Format
ax.set_title("Efficient Frontier - %s \n Cosine Similarity Estimates" %industry_name)
ax.legend()
plt.tight_layout()
plt.savefig('images/Efficient_Frontier_Cosine_Similarity_Estimates' + str(industry_name) + '.png', dpi=200, bbox_inches='tight')
plt.show()


# ##### Min Volatility Portfolio

# ###### Performance

# In[25]:


ef2.portfolio_performance(verbose=True);


# ###### Weights

# In[26]:


companies = []
weights = []
for company, weight in ef2.clean_weights().items():
    if weight != 0:
        companies.append(company)
        weights.append(weight)
        
dic = {'Company_Name':companies,'Weight':weights}
min_vol = pd.DataFrame(dic)
min_vol.to_csv("data/min_vol_cos_sim_Real_Estate_Investment_Trusts.csv", index = False)


# In[27]:


pd.read_csv("data/min_vol_cos_sim_Real_Estate_Investment_Trusts.csv")


# #### State Commercial Banks (commercial banking)

# In[28]:


SIC = SIC_list[3]
    
industry_name = SIC_desc[SIC_list.index(SIC)]
    
# get the names of the companies in the other industries
Companies = df[df.SIC == SIC]
Company_list = Companies.index

# get the companies name that match return data and business description data
SET = (set(Company_list) & set(r_selected.index))
LIST = [*SET, ]

mu_sample = mu[LIST]
# get the outliers
outlier = mu_sample[mu_sample>1].index
mu_sample = mu_sample.drop(outlier)
LIST = mu_sample.index

tmp = cov[LIST].T
cov_sample = tmp[LIST]

tmp = cosine_sim[LIST].T
tmp = tmp[~tmp.index.duplicated(keep="first")]
industry_cos_sim = tmp[LIST].T
industry_cos_sim = industry_cos_sim[~industry_cos_sim.index.duplicated(keep="first")]

sd = pd.DataFrame(np.sqrt(np.diag(np.diagonal(cov_sample))))
sd = sd.set_index(cov_sample.index)
sd.columns = cov_sample.index
cos_sim_cov = pd.DataFrame((np.dot(np.dot(sd, industry_cos_sim),sd)))

# perform minimum variance analysis
ef1 = EfficientFrontier(mu_sample, cos_sim_cov, weight_bounds=(0, 0.2))

fig, ax = plt.subplots()
plotting.plot_efficient_frontier(ef1, ax=ax, show_assets=True)

# Find and plot the tangency portfolio
ef2 = EfficientFrontier(mu_sample, cos_sim_cov, weight_bounds=(0, 0.2))
# min volatility
ef2.min_volatility()
ret_tangent, std_tangent, _ = ef2.portfolio_performance()
ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Min Volatility")

# Format
ax.set_title("Efficient Frontier - %s \n Cosine Similarity Estimates" %industry_name)
ax.legend()
plt.tight_layout()
plt.savefig('images/Efficient_Frontier_Cosine_Similarity_Estimates' + str(industry_name) + '.png', dpi=200, bbox_inches='tight')
plt.show()


# ##### Min Volatility Portfolio

# ###### Performance

# In[29]:


ef2.portfolio_performance(verbose=True);


# ###### Weights

# In[30]:


companies = []
weights = []
for company, weight in ef2.clean_weights().items():
    if weight != 0:
        companies.append(company)
        weights.append(weight)
        
dic = {'Company_Name':companies,'Weight':weights}
min_vol = pd.DataFrame(dic)
min_vol.to_csv("data/min_vol_cos_sim_State_Commercial_Banks.csv", index = False)


# In[31]:


pd.read_csv("data/min_vol_cos_sim_State_Commercial_Banks.csv")

