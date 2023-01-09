# ThemeBERT-UDA
UDA Final Project Repo



All the work have been done independently. 



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

- **Section II: Modelling & Research** Implement the research and reproduce the findings as is shown in the report. 

- **Section III: Practical Application**: Generate the Covid-19 news theme trend

  

## Running Time

All the analysis has been ran on local desktop (i.e., Apple M1 Macbook Pro of 32 GB memory) using single CPU. The runtime takes 1-2 hours on each dataset and no parallel jobs are needed. 

To efficiently rerun the analysis, user could skip some processing component and directly load the intermediate study variable. This could help save time to within 1 hour.

* A. Results on Real Financial News Dataset

  ```python
  topic_TC_final_result = joblib.load("study/findings/news_topic_TC.joblib")
  topic_TD_final_result = joblib.load("study/findings/news_topic_TD.joblib")
  
  ```

* B. Results on 20 Newsgroup Dataset

  Run the `Topic Model Comparison on Benchmark Dataset: 20 Newsgroups` block from scratch



# Contact Information

Email Address: [xiaohui.li21@imperial.ac.uk](mailto:xiaohui.li21@imperial.ac.uk) 
