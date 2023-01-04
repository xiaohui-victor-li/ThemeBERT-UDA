# Transformers NN Engine 
# ----------------------------
# -latest @12/14/2022 
#
# based on HuggingFace Transformers
#

from tqdm import tqdm
from typing import List, Union

import numpy as np
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from transformers.modeling_outputs import BaseModelOutput

from ._base import BaseEmbedder


### Model Directory
#
TRANSFORMER_MODEL_DIR_CONFIG = {
    'E5-small': '/mnt/ebs1/data/Victor/NLP/model/E5'
}

class TransformerEmbedder(BaseEmbedder):
    """ Transformers text embedding model

    The sentence-transformers embedding model used for generating document and
    word embeddings.

    Arguments:
        embedding_model: A sentence-transformers embedding model

    Usage:

    To create a model, you can load in a string pointing to a transformers model:
    """
    def __init__(self, embedding_model: Union[str, AutoModel] = "E5-small"):
        super().__init__()

        if isinstance(embedding_model,AutoModel):
            self.embedding_model = embedding_model
        elif isinstance(embedding_model, str):
            model_path = TRANSFORMER_MODEL_DIR_CONFIG[embedding_model]
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.embedding_model = AutoModel.from_pretrained(model_path)
        else:
            raise ValueError("Please select a correct HugginifaceTransformers model")
    
    def embed(self,
              documents: List[str],
              batch_size: int = 128,
              add_special_tokens: bool = False,
              verbose: bool = False) -> np.ndarray:
        """Embed a list of n documents/words into an n-dimensional matrix of embeddings

        Arguments:
            documents: A list of documents or words to be embedded
            batch_size: number of batch size to process the docs
        """
        doc_len = len(documents)

        # simple processing
        if doc_len<=batch_size:
            batch_dict = self.tokenizer(documents, max_length=512,
                                        padding=True,
                                        truncation=True,
                                        add_special_tokens=add_special_tokens,
                                        return_tensors='pt')
            
            hidden_outputs = self.embedding_model(**batch_dict)

            embed_outputs = self._average_pool(hidden_outputs.last_hidden_state,
                                               batch_dict['attention_mask']).detach().numpy()
        else: 
            # Batch-wise processing
            embed_batch_ls = []

            for start_idx in tqdm(range(0, doc_len, batch_size)):
                input_doc_ls = documents[start_idx:(start_idx+batch_size)]

                batch_dict = self.tokenizer(input_doc_ls, max_length=512, padding=True,
                                            truncation=True,
                                            add_special_tokens=add_special_tokens,
                                            return_tensors='pt')

                hidden_outputs = self.embedding_model(**batch_dict)

                embed_outputs_batch = self._average_pool(hidden_outputs.last_hidden_state,
                                                         batch_dict['attention_mask']).detach().numpy()
                
                embed_batch_ls.append(embed_outputs_batch)
            embed_outputs = np.vstack(embed_batch_ls)

        return embed_outputs
        
    # helper functions
    def _average_pool(self,
                      last_hidden_states: Tensor,
                      attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
