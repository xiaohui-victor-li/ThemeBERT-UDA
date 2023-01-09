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

  
  Evaluation Results: `study/findings/{news_topic_TC.joblib|news_topic_TD.joblib}`
  
  

* FinNLP Python Library

  An open-source NLP library `FinNLP` is developed by myself, which is extended from open source codes. The library supports the functionality of topic model ThemeBERT and other NLP models.  

  ⚠️ The following python libraries are required to pre-install to launch `FinNLP`:

  ```python
  SpaCy: https://spacy.io/usage
  PyTorch: https://pytorch.org/get-started/locally/
  HuggingFace Transformers [torch]: https://huggingface.co/docs/transformers/installation
  Sentence-BERT: https://www.sbert.net/
  UMAP-learn: https://umap-learn.readthedocs.io/en/latest/
  Hdbscan: https://hdbscan.readthedocs.io/en/latest/ 
  rich: https://pypi.org/project/rich/
  
  ```

  

# Quick Start to Reproduce Results

To run the analysis, open the `/ThemeBERT/ThemeBERT_Study.ipynb` analytics notebook. All the analysis are commented with guidance.

The notebook analysis in total consists of three sections:

- **Section I: Data Processing.** Collect the real global financial news text data using selenium and Wayback Machine Python. The data collection process is *independent* to the analysis process. Could directly load the generated data `fin_news_studydata.parquet`.

  ```python
  news_data_output = pd.read_parquet("study/fin_news_studydata.parquet")
  
  # research data
  news_research_data = news_data_output.sample(10000, random_state=42).copy()
  
  # text document input
  input_doc_ls = news_research_data['NewsInput'].to_list()
  
  ```

  

- **Section II: Modelling & Research.** Implement the research and reproduce the findings as is shown in the report. 

  

- **Section III: Practical Application.** Generate the Covid-19 news theme trend

  Run the notebook code of the `Section III` block. The output will be the time series charts of the Covid-19 news trend tracker.

  

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
