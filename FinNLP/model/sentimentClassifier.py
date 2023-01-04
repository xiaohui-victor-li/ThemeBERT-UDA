# Sentiment Classificaiton Module
# --------------------------------
# -@latest 12/14/2022;
#

from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
from scipy.special import softmax

NLP_MODEL_DIR_CONFIG = {
    'analyst-tone': '/mnt/ebs1/data/Victor/NLP/model/finbert-analyst-tone',
    'news-sentiment': '/mnt/ebs1/data/Victor/NLP/model/finbert-news-sentiment',
    'forward-looking': '/mnt/ebs1/data/Victor/NLP/model/finbert-FL'
}
NLP_TOKEN_DIR_CONFIG = {
    'analyst-tone': '/mnt/ebs1/data/Victor/NLP/model/finbert-analyst-tone/token-tone',
    'news-sentiment': '/mnt/ebs1/data/Victor/NLP/model/finbert-news-sentiment/token-tone',
    'forward-looking': '/mnt/ebs1/data/Victor/NLP/model/finbert-FL/token-tone'
}

class TransformersSentimentEngine:
    """Sentiment Classifciaton Engine

    Core Function:
        @identify_sentiment
    """
    def __init__(self, model='analyst-tone', num_labels=3, model_dir=None, token_dir=None, device=None, verbose=True):
        """
        """
        self.model_type = model
        if model_dir is None:
            model_dir = NLP_MODEL_DIR_CONFIG[model]
        if token_dir is None:
            token_dir = NLP_TOKEN_DIR_CONFIG[model]
        self.token_dir = token_dir
        self.model_dir = model_dir
        self.num_label = num_labels
        self.device = device  # device to use for pytorch model: cude or cpu
        self.verbose = verbose

        # neural network model
        if self.verbose:
            print(" -> Loading the model [{}] at {}".format(self.model_type, model_dir))

        
        ## * customized model attributes
        self._load_model_engine()

        if self.model_type == 'analyst-tone':  ## analyst tone FinBERT of multiple labels
            self.labels_map = {0:'neutral', 1:'positive',2:'negative'}
        elif self.model_type == 'news-sentiment':  ## news sentiment of multiple labels
            self.labels_map = {0:'negative', 1:'neutral',2:'positive'}
        elif self.model_type == 'forward-looking':  ## news sentiment of multiple labels
            self.labels_map = {0:'notFL', 1:'nonspecificFL',2:'specificFL'}

    def _load_model_engine(self):
        if self.model_type in ['analyst-tone', 'news-sentiment', 'forward-looking']:
            self.model_engine = BertForSequenceClassification.from_pretrained(self.model_dir,
                                                                              num_labels=self.num_label)
            self.tokenizer = BertTokenizer.from_pretrained(self.token_dir)

        if self.device is not None:
            self.model_engine.to(self.device)
        self.device = self.model_engine.device  # update the model device status
        if self.verbose:
            print(" ==> Model engine is activated successfully --using device [{}]!".format(self.device))

    def preprocess(self, inputs, padding=True):
        """Preprocess the input text list

        Input Arg:
            input: a list of string text of size B
        """
        input_processs = self.tokenizer(inputs,
                                        return_tensors="pt",
                                        padding=padding)
        return input_processs

    def forward(self, tensor_inputs):
        """Call the Transformer model to generate output

        Output:
            sentiment prediction : a tensor of size BxM
        """
        tensor_output = self.model_engine(**tensor_inputs)[0]

        return tensor_output 

    def identify_sentiment(self, inputs, padding=True, label_names = None):
        """Directly output the sentiment probability as pandas

        Input Arg:
            input: a list of string text of size B

        Output:
            sentiment probability prediction : a pandas DF of size BxM
        """
        input_tensor = self.preprocess(inputs, padding=padding)

        # enable GPU CUDA for input data
        if self.device is not None and self.device.type != 'cpu':
            input_tensor = input_tensor.to(self.device)
        logit_tensor = self.forward(input_tensor)

        prob_output = logit_tensor.detach().cpu().numpy()
        if label_names is None:
            label_names = self.labels_map
        prob_output_df = pd.DataFrame(prob_output).rename(columns=self.labels_map)
        # normalize to sum to one across row
        prob_output_df = softmax(prob_output_df, axis=1)

        return prob_output_df