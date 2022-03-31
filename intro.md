# Unsupervised analysis of Textual data from 10K Financial Reports

In this research report we will be diving into Annual Reports of publicly traded companies. Specifically, we will analyze the text in the `Business Description` section of each report.

## Company Embeddings
The report will first go through techniques such as `tf-idf` and `word2vec` in an attempt to to create document embeddings in an `n` dimensional space. The goal is to find techniques that cluster company filings closer if those companies are very related, and farther if they are unrelated. We used their **SIC** category (industry category) as our reference variable to evaluate our results.

## Topic Modelling
Topics created based on the words available in the corpus theoretically resemble the respective industries each company operates in. This section examines the possibility to extract topics from the corpus of annual reports, potentially allowing us to create our own groupings for companies such as _software_ or _pharmaceutical_. This would be another avenue we could take to explore clustering companies based on their relationship to all the topics.

To extend the previous idea,  multiple annual reports per company were analyzed to understand how companies may have changed in business direction or target industries over multiple years.

## Text to Financial Return Relationship
Lastly, the relationship between companies based on the document embeddings and their financial returns was analyzed. Monte-Carlo simulations we're used to analyze different portfolios of companies in a given industry and comparing the returns of those simulations to the Efficient Frontier.

Check out each page bundled within this book to see more on a given topic.

```{tableofcontents}
```
