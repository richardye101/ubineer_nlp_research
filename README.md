<!-- _Cloned from [Jupyter Lite Demo](https://github.com/jupyterlite/demo)_ -->

Checkout our analysis on our website: https://richardye101.github.io/ubineer_nlp_research/

# NLP Research into public 10K filings

In this project, Mary Xu, Peitong Lu, and Richard Ye worked with the `business description` of 1500+ 10K Annual Report Filings from SEC, with support from [Ubineer](https://www.ubineer.com/).

## Topics explored

1. Word Embedding techniques
2. Distance metrics on those word embeddings
3. Topic Modelling/Emebedding
4. Dynamic Word Embedding comparisons

## Data cleaning/processing

The data was provided to us by Ubineer, which has been pulled and preprocessed for us. We further cleaned up the `Description` text by removing HTML code, the first couple words which were common among all filings, and filtering for filings with over 250 characters.

We then removed _stop words_ from the business descriptions, which are very commong words like "the, they, and, is, are, there" and others. These words don't provide meaning and therefore do not contribute to our goal of extracting meaning.

We also lemmatized all possible words, aka Text/Word Normalization which means all instances of "am, are, is" are converted to "be" and "playing, played, plays" are all converted to "play". This reduces the amount of different words we have to process, and also condensing the amount of information we recieve since words that all carry the same meaning are represented together.

## Evaluation and visualization techniques

In this project we focused on visually examining 2 and 3 dimensional plots of the word embeddings reduced using PCA and Truncated SVD (specifically for LSA). We also had access to extra information for 2018 filings, allowing us to evaluate the word embedding clusters against their actual industry classification. This was done using a simple 1-NN clustering. The results were put into a confusion matrix, allowing us to identify how well the 1-NN clutsering did on our word embedding.

# 1. Word Embeddings techniques

## Term Frequency/Counter Vectorizer/Bag of Words

We started off with the basic Term Frequency Matrix, which breaks down each company description into a vector of `n` words/terms (a hyperparameter), where each dimension is a word/term, and the value is the count of that word in the document.

This technique helps us analyze how many of the `n` words eaach filing contains, which provides us with information about the kind of terms or topics each company may discuss. This approach is very easy to implement but is not very powerful, because very common words will have the largest values and therefore carry the most weight. For financial statements like these, you can expect words like "financial" and "report" to have some of the highest values.

From these vectors for each company filing, we can think of each term as a dimension and actually project these `n` dimensional vectors into an `n` dimensional space, which is called a **word embedding**. You can think of these as points in a `n`D space.

## Term Frequency - Inverse Document Frequency (tf-idf)

To solve the above issue, we moved on to the tf-idf technique. tf-idf augments the term frequency matrix we created above by multiplying each word in each docuemnt by its "importance" to that document. The details are within the [1_Tf-idf_analysis.ipynb](https://github.com/richardye101/ubineer_nlp_research/blob/main/content/richard/1_Tf-idf_analysis.ipynb) notebook. This technique is meant to adjust the weighting of terms used in the word embedding so that the _points_ used to represent each company filing is more accurate in representing where companies are in this `n` dimensional space in comparison to other companies. For example, ideally we want technology companies close together, and pharmaceutical companies close together.

## 


