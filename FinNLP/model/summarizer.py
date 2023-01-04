# Text Summary Generation Module
# ------------------------------------
# -latest@12/94/2022; 
#

from transformers import pipeline
from tqdm import tqdm 

import pandas as pd

NLP_MODEL_DIR_CONFIG = {
    'BARTnews-summarizer': '/Users/lxh/Desktop/Fintelligence/FintelDirectory/data/model/BART-CNN-summarizer',
    'T5small-summarizer': '/mnt/ebs1/data/Victor/NLP/model/T5small-summarizer'
}

class TransformerSummarizer:
    """Summarization text generator
    """
    def __init__(self, model='BARTnews-summarizer', model_dir=None, verbose=True,
                 flag_hard_max_length=False,  # whether to enforce strict max length constrain
                 device='cpu',
                 max_length=None,
                 min_length=24):
        self.model_type = model
        self.device = device
        if model_dir is None:
            model_dir = NLP_MODEL_DIR_CONFIG[model]
        self.model_dir = model_dir
        self.verbose = verbose
        self.max_length_constraint = flag_hard_max_length
        self.max_length = max_length
        self.min_length = min_length
        # launch the pre-trained summary model
        self._load_model_engine()
    
    def _load_model_engine(self):
        if self.device=='gpu'or self.device=='cuda':
            self.model_engine =  pipeline("summarization",
                                          model=self.model_dir,
                                          device=0,   # use the default GPU engine
                                          framework='pt')
        else:
            self.model_engine =  pipeline("summarization",
                                          model=self.model_dir,
                                          framework='pt')

    def summarize_text(self, docs, batch_size=64,
                       max_length=None, min_length=None, show_progress_bar=True, truncation=True):
        """

        Input Arg:
            docs: a list of text context or single str
        """
        # convert into list
        if type(docs) != list:
            docs = [docs]
        # summary max length for all input docs
        if max_length is None:
            global_max_length = self.max_length
        else:
            global_max_length = max_length

        # init the summary obj
        summary_docs = []
        disable_progress_bar = not show_progress_bar

        if batch_size is None:
            ## mode of single summarization
            for doc in tqdm(docs, disable=disable_progress_bar):
               
                max_length_bound = global_max_length
                min_length_bound = self.min_length
                ## call @summarizer
                sum_doc = self.model_engine(doc, #max_length=max_length_bound,
                                            min_length=min_length_bound,
                                            truncation=truncation,
                                            do_sample=False) 

                summary_docs.append(sum_doc[0])
        else:
            ## mode of bacth-level summary
            max_length_bound = global_max_length

            if min_length is None:
                min_length_bound = self.min_length
            else:
                min_length_bound = min_length
           
            for idx in tqdm(range(0, len(docs), batch_size)):
                start_idx, end_idx = idx, idx+batch_size 
                batch_doc_ls = docs[start_idx:end_idx]
                ## call @summarizer for the batch
                sum_docs = self.model_engine(batch_doc_ls,
                                             #max_length=max_length_bound,
                                             min_length=min_length_bound,
                                             truncation=truncation,
                                             do_sample=False) 
                summary_docs.extend(sum_docs)



        summary_df = pd.DataFrame(summary_docs)

        return summary_df

