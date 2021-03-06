## <center>Unsupervised analysis of Textual data from 10K Financial Reports</center>

### <center>(STAD95 Project Report)</center>

## Summary

- Word Embedding results
  - We evaluated several models
  - We were able to accomplish up to 94% recall for predicting similar companies when matched with their categories.
- Topic Modelling results
  - can reveal significant changes but increase in an topic does not necessarily mean the company is leaning more in the direction of the topic.
  - Increase or decrease in topic could bring investor attention to the topic at hand for more investigation
- Portfolio selection results

## Introduction

Public-traded companies file a comprehensive annual financial report(10K) to discuss their financial performance as required by the U.S Securities and Exchange Commission(SEC). These reports can contain quantitative data and qualitative data. Quantitative data includes the income statement, balance sheets, and statement of cash flows. Qualitative data includes a description of the business, risk factors, and management's discussion and analysis. Many researchers and investors have saturated the area of using quantitative data to build portfolios and evaluate the risks and returns of a company. However, there has not been much focus in the analysis of qualitative data provided in financial reports. In this report, we hope to focus our attention on the analysis of qualitative data which provides more forward-looking information that may reveal the company's plans and anticipated events/risks.

This study focuses on the analysis of the Business section of the 10K filings. The Business section provides an overview of the company's main operations, including its products and services. It may also include recent events, competition, regulation, labor issues, operating costs, or insurance matters.

Now we provide a short overview of our approach. Our work is considered textual analysis on unstructured data(data not organized in a pre-defined manner) since individual companies may include different sections of text in the business description of a 10k report. We apply various document embedding techniques to model the similarity between companies using the corresponding business description (i.e. the document in discussion). We then branch off into two separate directions. First. we do a deep dive into each individual company using topic modelling techniques to understand the general categories associated with their business model. We also apply topic modelling in the dynamic analysis of emerging trends within a company over a number of years. In this work, we specifically use Netflix and General Electric as a proof of concept to validate our hypothesis that there should be a significant change in a topic category during a specific year that a company has altered their business model. Second, we perform mean-variance analysis and construct portfolios for each industry using the document embedding results to explore the relationship between returns and business description. Minimum variance portfolios are constructed based on three estimates. The three estimates are simple sample covariance, cosine similarity generated covariance and factor model generated covariance. The portfolio using simple sample covariance is considered as reference to the other two portfolios to determine the feasibility of constructing optimized portfolios with the word embedding results. 

This study is exploratory and so our desired outcome is to provide insight into the multitude of unstructured textual analysis techniques and their fitness for our data and purpose.

## Methodology

### Data

We used the business description(ie. Section 1) of 10K Annual Report Filings from SEC, with support from Ubineer for extracting the necessary data. Additionally, we joined our dataset with SIC codes from the EDGAR database to obtain information regarding the SIC category of the company to better evaluate our results on company similarities and differences.

The 2018 filings of companies from the top 5 categories (Prepackaged Software, Pharmaceutical Preparations, Crude Petroleum and Natural Gas, Real Estate Investment Trusts, State Commercial Banks) were used to train our word embedding models. There were a total of 1127 filings before we applied preprocessing techniques and 618 filings after.

The filings of companies from 2016 to 2018 were used in the dynamic topic analysis models. There were a total of 2008 filings before we applied preprocessing techniques and 1692 after.

The monthly stock returns data from the top 5 SIC industries was extracted from Wharton Research Data Services???s (WRDS) CRSP/Compustat database. After grouping the returns data in terms of company name and date, removing duplicate values, and selecting only from June 2016 to December 2018, there were a total of 719 companies with 31-month returns.

### Methods

The research design of this study was experimental and exploratory.

#### Company Embeddings

We started by investigating a variety of word embedding to model the description similarity between companies. These modeling techniques include TF-IDF, Word2Vec, Universal Sentence Encoder, Doc2Vec, and many others that we will discuss in detail in this report. Prior to fitting each model, we applied consistent data preprocessing methods such as word normalization by lemmatizing, stop word removal, and special character removal. We also applied dimensionality reduction using PCA and truncated SVD (specifically for LSA) to visualize the 2D and 3D plots of word embeddings. We then evaluated the performance of our embedding models by comparing the cosine distance between text embeddings and applying a 1-nearest-neighbor algorithm to determine the accuracy of SIC category predictions. The goal is to find techniques that cluster company filings closer if those companies are very related, and farther if they are unrelated.

#### Topic Modelling

Topics created based on the words available in the corpus theoretically resemble the respective industries each company operates in. This section examines the possibility to extract topics from the corpus of annual reports, potentially allowing us to create our own groupings for companies such as _software_ or _pharmaceutical_. This would be another avenue we could take to explore clustering companies based on their relationship to all the topics. We investigated topic modeling techniques such as Latent Dirichlet Allocation, Non-Negative Matrix Factorization, and Latent Semantic Analysis. We manually interpreted and evaluated the performance of the topic models as these techniques are known to be difficult to perform quantitative evaluations for.

To extend the previous idea, multiple annual reports per company were analyzed to understand how companies may have changed in business direction or target industries over multiple years.

#### Text to Financial Return Relationship

Lastly, the relationship between companies based on the document embeddings and their financial returns was analyzed. Monte-Carlo simulations were used to analyze different portfolios of companies in a given industry and compare the returns of those simulations to the Efficient Frontier. For the data collection, we used the company's Central Index Key (CIK) as the identifier to obtain monthly stock returns data from Wharton Research Data Services???s (WRDS) CRSP/Compustat database. We included only the companies with filings that were used in our previous embedding work and the timespan of returns ranges for each filer, but they are all from June 2016 to December 2018. We conducted a mean-variance analysis with three covariance estimates - simple sample covariance, cosine similarity generated covariance, and factor model generated covariance, to construct minimum-variance portfolios. As for cosine similarity estimate, we use the cosine similarity distance as correlation and sample return standard deviation to calculate the covariance estimate. In terms of factor model estimate, we first conduct Sent-LDA model on company???s business description to build a matrix. Then, we ran linear regression on the topic model matrix with the returns matrix to get the coefficient matrix. Lastly, we calculate the factor model estimated covariance using topic model matrix, covariance of coefficient matrix and the diagonal matrix of residual variance.

#### Data Preprocessing

The data was provided to us by Ubineer, which has been pulled and preprocessed for us. One of the main datasets we used is the `bq_2018_top5SIC.json` file prepared by Professor Sotiros Damouras, by selecting companies who have filed in 2018 and belong to the top 5 industries within the dataset. This file has 1127 filings (one per company).

The file schema contains the columns:

- `accessionNumber`
- `filingDate`
- `reportingDate`
- `financialEntity`
- `htmlFile`
- `coDescription`
- `CIK`
- `name`
- `countryinc`
- `cityma`
- `SIC`
- `SIC_desc`

For our purposes, we will be focusing on `name` (identifies the company), `coDescription` (the Business Overview), `SIC_desc` (the industry they operate in)

Within our pre-processing, we focus on `coDescription`.
We further cleaned up the `Description` text by removing HTML code, the first couple words which were common among all filings such as _"business overview"_, and filtering for filings with over 250 characters.

We then removed _stop words_ from the business descriptions, which are very commong words like "the, they, and, is, are, there" and others. These words don't provide meaning and therefore do not contribute to our goal of extracting meaning.

We also lemmatized all possible words, aka Text/Word Normalization which means all instances of "am, are, is" are converted to "be" and "playing, played, plays" are all converted to "play". This reduces the amount of different words we have to process, and also condensing the amount of information we recieve since words that all carry the same meaning are represented together.

Check out each page bundled within this book to see more on a given topic.

```{tableofcontents}

```
