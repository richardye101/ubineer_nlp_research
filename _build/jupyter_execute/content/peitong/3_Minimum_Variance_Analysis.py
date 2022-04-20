#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.insert(0, '..')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('aimport', 'std_func')


# ## Mean-Variance Analysis - Minimum Variance
# 
# Mean-variance analysis is a mathematical framework that examplifies the trade-off between return and risk. It is used to create diversified portfolios based on investors’ expectation. There are one main approach used in this report. We have the minimum volatility portfolio that concentrates on minimizing the risk of the portfolio. Mimimum variance portfolio can help us compare the correlation of simple sample covariance, covariance generated using cosine similarity distances and covariance generated using factor model in Sent-LDA.

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ### Monthly Returns
# Since we will generate cosine similarity estimate and factor model estimate using business description of companies from 2016 to 2018, we only consider monthly returns before 2019. We group the monthly stock returns data extracted from Wharton Research Data Services’s (WRDS) CRSP/Compustat database by company name and date first. After a selection of time span, from June 2016 to December 2018, we are able to get 31-month returns for 719 companies.

# In[3]:


r_selected = pd.read_csv("data/filtered_r.csv")
# get the mean of all 
r_selected.set_index("name", inplace = True)
mu = r_selected.mean(axis = 1)
# compute the covariance matrix 
cov = r_selected.T.cov()


# In[4]:


r_selected


# #### Sample Mean

# In[5]:


mu


# #### Sample Covariance

# In[6]:


cov

