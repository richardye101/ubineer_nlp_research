#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

def highlight(val):
    color = 'blue' if val in Same else 'black'
    return 'color: %s' % color


# ## Porfolio Analysis Results
# In this section, the porfolio weight table for each industry in terms of three kinds of estimates are displayed, and the companies that exist in all three constructed portfolios are hilighted.

# In[2]:


columns = pd.MultiIndex.from_product([["Prepackaged Software", 
                                       "Pharmaceutical Preparations", "Real Estate Investment Trusts", 
                                       "State Commercial Banks",],
                                      ['Sample', 'Cosine Similarity', 'Factor Model']])

data = [[0.6,1.1,2.1,1.2,1.2,0.4,0.5,0.6,0.3,1.2,1.1,0.9],
        [2.4,2.9,15.9,2.1,2.6,12.2,1.8,1.7,8.0,2.7,2.2,20.8],
        [-0.57,-0.30,0.00,-0.35,-0.32,-0.13,-0.80,-0.81,-0.22,-0.28,-0.38,-0.05]]

methods = ["Expected Annual Return", "Annual Volatility", "Sharpe Ratio"]


import pandas as pd
df = pd.DataFrame(data, index = methods, columns = columns).T.round(2)

import seaborn as sns
cm = sns.light_palette("#5CCDC6", n_colors = 35, as_cmap=True)

df.style.background_gradient(cmap=cm)


# ### Prepackaged Software (mass reproduction of software)

# In[3]:


sample_software = pd.read_csv("data/min_vol_sample_Prepackaged_Software.csv")
cos_sim_software = pd.read_csv("data/min_vol_cos_sim_Prepackaged_Software.csv")
factor_model_software = pd.read_csv("data/min_vol_factor_model_Prepackaged_Software.csv")

sample_software = sample_software.sort_values(by=["Weight"], ascending=False).reset_index(drop=True)
cos_sim_software = cos_sim_software.sort_values(by=["Weight"], ascending=False).reset_index(drop=True)
factor_model_software = factor_model_software.sort_values(by=["Weight"], ascending=False).reset_index(drop=True)

columns = pd.MultiIndex.from_product([['Sample Estimate', 'Cosine Similarity Estimate', 'Factor Model Estimate'], 
                                      ['Company Name', 'Weight']])
software = pd.concat([pd.concat([sample_software, cos_sim_software], axis=1), factor_model_software], axis=1)
software.columns = columns

Same = (set(sample_software.Company_Name) & set(cos_sim_software.Company_Name)) & set(factor_model_software.Company_Name)


# In[4]:


software.style.applymap(highlight)


# ### Pharmaceutical Preparations

# In[5]:


sample_pharm = pd.read_csv("data/min_vol_sample_Pharmaceutical_Preparations.csv")
cos_sim_pharm = pd.read_csv("data/min_vol_cos_sim_Pharmaceutical_Preparations.csv")
factor_model_pharm = pd.read_csv("data/min_vol_factor_model_Pharmaceutical_Preparations.csv")


# In[6]:


sample_pharm = sample_pharm.sort_values(by=["Weight"], ascending=False).reset_index(drop=True)
cos_sim_pharm = cos_sim_pharm.sort_values(by=["Weight"], ascending=False).reset_index(drop=True)
factor_model_pharm = factor_model_pharm.sort_values(by=["Weight"], ascending=False).reset_index(drop=True)


# In[7]:


columns = pd.MultiIndex.from_product([['Sample Estimate', 'Cosine Similarity Estimate', 'Factor Model Estimate'], 
                                      ['Company Name', 'Weight']])
pharm = pd.concat([pd.concat([sample_pharm, cos_sim_pharm], axis=1), factor_model_pharm], axis=1)
pharm.columns = columns
Same = (set(sample_pharm.Company_Name) & set(cos_sim_pharm.Company_Name)) & set(factor_model_pharm.Company_Name)


# In[8]:


pharm.style.applymap(highlight)


# ### Real Estate Investment Trusts

# In[9]:


sample_real_estate = pd.read_csv("data/min_vol_sample_Real_Estate_Investment_Trusts.csv")
cos_sim_real_estate = pd.read_csv("data/min_vol_cos_sim_Real_Estate_Investment_Trusts.csv")
factor_model_real_estate = pd.read_csv("data/min_vol_factor_model_Real_Estate_Investment_Trusts.csv")


# In[10]:


sample_real_estate = sample_real_estate.sort_values(by=["Weight"], ascending=False).reset_index(drop=True)
cos_sim_real_estate = cos_sim_real_estate.sort_values(by=["Weight"], ascending=False).reset_index(drop=True)
factor_model_real_estate = factor_model_real_estate.sort_values(by=["Weight"], ascending=False).reset_index(drop=True)


# In[11]:


columns = pd.MultiIndex.from_product([['Sample Estimate', 'Cosine Similarity Estimate', 'Factor Model Estimate'], 
                                      ['Company Name', 'Weight']])
real_estate = pd.concat([pd.concat([sample_real_estate, cos_sim_real_estate], axis=1), factor_model_real_estate], axis=1)
real_estate.columns = columns
Same = (set(sample_real_estate.Company_Name) & set(cos_sim_real_estate.Company_Name)) & set(factor_model_real_estate.Company_Name)


# In[12]:


real_estate.style.applymap(highlight)


# ### State Commercial Banks (commercial banking)

# In[13]:


real_estate.style.applymap(highlight)


# In[14]:


sample_banks = pd.read_csv("data/min_vol_sample_State_Commercial_Banks.csv")
cos_sim_banks = pd.read_csv("data/min_vol_cos_sim_State_Commercial_Banks.csv")
factor_model_banks = pd.read_csv("data/min_vol_factor_model_State_Commercial_Banks.csv")


# In[15]:


sample_banks = sample_banks.sort_values(by=["Weight"], ascending=False).reset_index(drop=True)
cos_sim_banks = cos_sim_banks.sort_values(by=["Weight"], ascending=False).reset_index(drop=True)
factor_model_banks = factor_model_banks.sort_values(by=["Weight"], ascending=False).reset_index(drop=True)


# In[16]:


columns = pd.MultiIndex.from_product([['Sample Estimate', 'Cosine Similarity Estimate', 'Factor Model Estimate'], 
                                      ['Company Name', 'Weight']])
banks = pd.concat([pd.concat([sample_banks, cos_sim_banks], axis=1), factor_model_banks], axis=1)
banks.columns = columns
Same = (set(sample_banks.Company_Name) & set(cos_sim_banks.Company_Name)) & set(factor_model_banks.Company_Name)


# In[17]:


banks.style.applymap(highlight)


# ### Crude Petroleum and Natural Gas
# Since there is no optimal portfolio generating from sample estimate and cosine similarity estimate for the Crude Petroleum and Natural Gas industry, we only display the portfolio weights for factor model estimate.

# In[18]:


factor_model_crude = pd.read_csv("data/min_vol_factor_model_Crude_Petroleum_and_Natural_Gas.csv")

sample_crude = sample_crude.sort_values(by=["Weight"], ascending=False).reset_index(drop=True)
factor_model_crude = factor_model_crude.sort_values(by=["Weight"], ascending=False).reset_index(drop=True)


# In[19]:


columns = pd.MultiIndex.from_product([['Cosine Similarity Estimate'], 
                                      ['Company Name', 'Weight']])

factor_model_crude.columns = columns


# In[20]:


factor_model_crude.style.applymap(highlight)

