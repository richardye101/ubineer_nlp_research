#!/usr/bin/env python
# coding: utf-8

# ##### Two Towers
# The two-tower model learns to represent two items of various types (such as user profiles, search queries, web documents, answer passages, or images) in the same vector space, so that similar or related items are close to each other. These two items are referred to as the query and candidate object, since when paired with a nearest neighbour search service such as Vertex Matching Engine, the two-tower model can retrieve candidate objects related to an input query object. These objects are encoded by a query and candidate encoder (the two "towers") respectively, which are trained on pairs of relevant items.
# 
# Since we wish to retrieve financial entities related to each other, we have the query item as the business description in plain text and the candidate item as the CIK ticker and its SIC category description of the financial entity. 
# 
# Training Data
# We used the 2018 filings of companies descriptions from the top 5 categories (Prepackaged Software, Pharmaceutical Preparations, Crude Petroleum and Natural Gas, Real Estate Investment Trusts, State Commercial Banks) to train our model. Filings from multiple years for each company is used so that the model has a better understanding of each company and its relevant descriptions. 
# 
# Structure of a single record in the training data: 
# ```
# {
#         "query":
#         {
#             "description": x["coDescription"]
#         },
#         "candidate":
#         {
#             "financial_entity": x["CIK"],
#             "category" : x["SIC_desc"]
#         }
# }
# ```

# ![twotowers_3d](../images/twotowers_3d.png)

# ![twotowers_cm](../images/twotowers_cm.png)
