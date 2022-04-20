#!/usr/bin/env python
# coding: utf-8

# # Our Setup: Data ETL and Methods of evaluation
# 
# This report relies upon the data which has analyzed and preprocessed in order to maximize information extraction. The original data was generously provided to us by [Ubineer](https://www.ubineer.com), a fin-tech start-up based out of the University of Toronto. This page will go in depth into the methods we used to obtain, clean, and preprocess our data. The methods used for plotting and evaluating the results are also described below.
# 
# ## The data
# 
# As mentioned above, we obtained our data from Ubineer, which they store in Googles BigQuery platform. They've already performed a lot of leg work processing gigabytes worth of annual reports, extracting text fields such as the `Business Description`, `Business Risk` and `Management's Discussion and Analysis of Financial Condition and Results of Operations`. In this report, we've focused on the `Business Description` data.
# 
# ### Schema
# 
# | Column          | Description                                                      |
# |-----------------|------------------------------------------------------------------|
# | accessionNumber | NA                                                               |
# | filingDate      | When the report was filed with the SEC                           |
# | reportingDate   | The date which the report was prepared for                       |
# | financialEntity | Contains the CIK, specifying which company the report belongs to |
# | htmlFile        | Link to the report                                               |
# | coDescription   | The Business Description section of the Report                   |
# 
# An example row of the data is below:
# 
# | accessionNumber      | filingDate              | reportingDate           | financialEntity                      | htmlFile                                                                           | coDescription                    |
# |----------------------|-------------------------|-------------------------|--------------------------------------|------------------------------------------------------------------------------------|----------------------------------|
# | 0001144204-09-017197 | 2009-03-31 10:22:32 UTC | 2008-12-31 05:00:00 UTC | financialEntities/params;cik=1140028 | https://www.sec.gov/Archives/edgar/data/1140028/000114420409017197/v143585_10k.htm | Item 1Business4<a href="#item1a" |
# 
# Notice how the Business Description column (`coDescription`) also contians some html noise.

# ### Extraction
# 
# Within the BigQuery interface, we were able to query the data, although due to the sheer size of it (2GB+) we were unable to exatract it efficiently for analysis. Our supervisor Professor [Sotirious Damouras](https://damouras.github.io/) was able to assist us in not only extracting the `coDescription` data, but also link each filing with a Company name, it's respective `SIC` code (identifying the company's operating industry), and the country and city they are headquartered. This data only contains filings from 2018, as the team agreed it would be best to avoid years plagued with COVID, but also have the most up to date information. As mentioned previously, the sheer size of data available prevented us from extracting all the company filings so we decided to filter only companies from the top five industries (based on number of companies). That gave us 1127 unique filings to analyze.
# 
# Here is a snippet:

# In[1]:


import pandas as pd
data = pd.read_json("../data/bq_2018_top5SIC.json", lines = True)
data.head()


# ### Cleaning
# 
# If you look closely, almost all `coDescription` value start with something like _"Item 1. Business Overview"_. Some even contain html code, identified by it's surrounding `<` and `>` angle brackets. One of the most important things to keep in mind is that our analysis can only be as good as our data. 
# 
# In an effort to improve our data, we'll be removing as much duplicate word data and HTML code as possible, as well as empty space and random quotations. This is done in the `std_func.py` files function `clean_data_fn()`, located [here](https://github.com/richardye101/ubineer_nlp_research/blob/5a16caf65a0d2d21e8f377bd8b5f0d6b2435ad84/content/std_func.py#L65-L92).
# 
# We also remove numbers, as they don't actually provide us with any semantic meaning.
# 
# As a final filter, we also remove any filings that contain less than 250 characters in the `coDescription` column, as they don't have enough data for us to analyze, or is a shell company with no actual business.
# 
# After cleaning, the `coDescription` column looks more like this:

# In[2]:


#update
import sys
sys.path.insert(0, '..')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('aimport', 'std_func')
cleaned_data = std_func.clean_data_fn(data)
pd.options.display.max_colwidth = 100
cleaned_data.head().loc[:,"coDescription"]


# In[3]:


print("After cleaning, there are {} filings remaining".format(cleaned_data.shape[0]))


# Much better!
# 
# ### Pre-processing
# 
# Now that the data has been cleaned, it's time to dig in and really analyze it. Through our analysis, we've found that many words such as "are" and "is", "running" and "ran" all carried their respective meaning, but yet represented as _different_ words. In order to condense the sheer amount of data we have (in number of words), we perform something called __lemmatization__, the process of reducing words to their base form. "are" and "is" would be converted to "be", and "running" and "ran" will be converted to "run". With less word variations to deal with, our analysis is bound to improve!
# 
# Another pre-processing step taken was removing stop words. These words include words such as "the", "and", "that", "is" among many more. These words themselves don't carry any meaning, and our goal is to extract as much semantic information as possible out of our data. As these stop words don't contriute to that goal (and they take up a LOT of room, just read the previou sentence and count the stop words!), we can further remove them to reduce the amount of data we need to process.
# 
# You can find the code [here](https://github.com/richardye101/ubineer_nlp_research/blob/5a16caf65a0d2d21e8f377bd8b5f0d6b2435ad84/content/std_func.py#L94-L114).
# 
# After these steps, our `coDescription` column looks like this:

# In[4]:


processed_data = std_func.clean(data)
processed_data.head().loc[:, "coDescription_stopwords"]


# This results in detailed, concise business descriptions with as much fluff removed as possible. Our analysis depends on having as much information as possible, while also reducing the extraneous bits that don't contribute to our analysis. This data is what we use through the majority of our analysis, existing as the `coDescription_stopwords` column.
