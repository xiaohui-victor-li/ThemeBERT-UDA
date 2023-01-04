# ThemeBERT-UDA
UDA Final Project Repo





# Overview

* Research Analysis Data

  Key research data are stored in GitHub at the following directory: ```/study```

  * 1. Created Global Financial News Dataset: `study/fin_news_studydata.parquet`

  * 2. Benchmark Dataset:

    ```python
    from sklearn.datasets import fetch_20newsgroups 
    
    newsgroups_train = fetch_20newsgroups(subset='train',
                                          remove=('headers', 'footers', 'quotes'))
    
    newsgroups_test = fetch_20newsgroups(subset='test',
                                         remove=('headers', 'footers', 'quotes'))
    
    ```

  		* Evaluation Results: `study/findings/{news_topic_TC.joblib|news_topic_TD.joblib}`



* FinNLP Python Library

  An open-source NLP library `FinNLP` is developed by myself, which is extend from open source codes. The library supports the functionality of topic model ThemeBERT and other models.  

  ⚠️ The following python libraries are required to pre-install:

  ```python
  SpaCy
  PyTorch
  HuggingFace Transformers
  Sentence-BERT
  Hdbscan | UMAP-learn
  rich: https://pypi.org/project/rich/
  
  ```

  



# Quick Start

To run the analysis, open the `/ThemeBERT/ThemeBERT_Study.ipynb` analytics notebook. All the analysis are commented with gudiance.

The notebook analysis in total consists of three sections:

- **Section I: Data Processing.** Collect the news text data using selenium and Wayback Machine Python. Could directly load the generated data.
- **Section II**: 

- **Section III: Practical Application**: Generate the Covid-19 news theme trend



# Contact Information

Email Address: [xiaohui.li21@imperial.ac.uk](mailto:xiaohui.li21@imperial.ac.uk) 
