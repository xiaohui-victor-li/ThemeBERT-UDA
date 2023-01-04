# 🌟Core Theme Idenfication NLP Class: ThemeBERT
# ------------------------
# -latest @12/18/2022; 
#
#   developed in progress [🔨]
#
#
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
from typing import List, Union, Tuple
from tqdm import tqdm

from packaging import version
from sklearn import __version__ as sklearn_version
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer

from FinNLP._utils import NLPLogger  # logger
from FinNLP.preprocess.keyphraseVectorizer import KeyphraseCountVectorizer
from FinNLP.model.summarizer import TransformerSummarizer

# ThemeBERT functionality
from ._mmr import mmr
from ._maxsum import max_sum_distance
from ._highlight import highlight_document
from .backend._utils import select_backend

# backend topicEngine class
from FinNLP.topicModel import TopicEngine

class ThemeBERT:
    """
    ThemeBERT: Contextual Theme Identification NLP Framework

    The keyword extraction is done by finding the sub-phrases in
    a document that are the most similar to the document itself.

    Document embeddings are extracted with sentence-BERT to get a
    document-level representation. 
    """
    def __init__(self,
                 vectorizer_spacy_pipeline="en_core_web_md",
                 model="all-MiniLM-L6-v2",  # embedding model
                 topic_engine_path=None,   # backend topic engine
                 n_gram_range=(1,2),
                 stop_words = None,
                 device='cpu',
                 umap_model=None,    # dimension reduction engine
                 summary_model=None,   # summarizer engine
                 verbose=True,
                 show_progress_bar=False,
                 flag_hard_max_length=False,  # whether to enforce max length
                 max_length=60,
                 nr_topics=None,  # number of topics to keep
                 min_topic_size=20  # topic size 
                 ):
        """KeyBERT initialization

        Input Args:
            model: Use a custom embedding model.
                   The following backends are currently supported:
                      * SentenceTransformers [default] - https://www.sbert.net/docs/pretrained_models.html
        """
        self.logger = NLPLogger("DEBUG")
        self.logger.info(" => 🚀Launching the ThemeBERT model with PLM embedding engine [{}] ...".format(model))
        self.device = device
       
        #### Spacy Processing Pipeline ####
        self.vector_spacy_pipeline = vectorizer_spacy_pipeline
        self.show_progress_bar = show_progress_bar

        if stop_words is None:
            self.stop_words = text.ENGLISH_STOP_WORDS
        else:
            self.stop_words = text.ENGLISH_STOP_WORDS.union(stop_words)

        #### Preprocessing&Embedding Module #### 
        ## set up the vectorizer
        # Entity-based 
        self.entity_vectorizer = KeyphraseCountVectorizer(spacy_pipeline=self.vector_spacy_pipeline,
                                                          stop_words=self.stop_words)
        # default
        self.vectorizer = None
        
        ## set up the summarizer
        self.hard_max_length = flag_hard_max_length
        self.sum_max_length = max_length
        if summary_model is None:
            self.summarizer = None
        else:
            self.summarizer = TransformerSummarizer(device=self.device,
                                                    model=summary_model,
                                                    flag_hard_max_length=self.hard_max_length,
                                                    max_length=self.sum_max_length)

        #### Theme Modelling Module ####

        # @TopicEngine as backend
        self.topic_engine_path = topic_engine_path
        if self.topic_engine_path is None:
            self.backend_model = TopicEngine(n_gram_range=n_gram_range,
                                             embedding_model=model,
                                             umap_model=umap_model, 
                                             min_topic_size=min_topic_size,
                                             nr_topics=nr_topics,
                                             verbose=verbose)
        else:
            self.logger.info(" ==> ⚙️ Load the stored backend TopicEngine --@{}".format(self.topic_engine_path))
            self.backend_model = TopicEngine.load(self.topic_engine_path,
                                                  embedding_model=model)
        # Contextual Embedding language model
        self.model = self.backend_model.embedding_model

        # UMAP - dimension reduction model
        self.umap_model = self.backend_model.umap_model

    ###### ✨API functionality ######
    def encode(
        self,
        docs: Union[str, List[str]],
        keyphrase_ngram_range: Tuple[int, int] = (1,2),
        batch_size=64,
        use_mmr=True,
        entity_threshold=0.25,
        semantic_threshold=0.3,
        flag_entity=True,
        flag_summary=True,
        flag_doc=False,
        show_progress_bar=True,
        truncation=True   # whether to truncate the document for summarization
        ):
        """core API function to encode document into list of embedding results

        Input Args:
            docs: The document(s) for which to extract keywords/keyphrases
        """
        output_ls = []   # init output result: a list of the dictionary

        ## Step 1. keyphrases
        if flag_entity:
            entity_themes = self.extract_keywords(docs,
                                                keyphrase_ngram_range=keyphrase_ngram_range,
                                                use_mmr=use_mmr,
                                                stop_words=self.stop_words,
                                                vectorizer='entity',
                                                show_progress_bar=show_progress_bar)
            
        semantic_themes = self.batch_extract_keywords(docs,
                                                      keyphrase_ngram_range=keyphrase_ngram_range,
                                                      batch_size=batch_size,
                                                      use_mmr=use_mmr,
                                                      stop_words=self.stop_words,
                                                      show_progress_bar=show_progress_bar)

        ## Step 2. Summary
        if flag_summary:
            if self.summarizer is None:
                self._load_summarizer()
            sum_doc_df = self.summarizer.summarize_text(docs,
                                                        batch_size=batch_size,
                                                        show_progress_bar=show_progress_bar,
                                                        truncation=truncation)

        ## Final Stage: Aggregation
        if isinstance(semantic_themes[0], tuple):
            semantic_themes = [semantic_themes]

        for idx,_ in enumerate(docs):
            # include keyphrases
            phrase_i = semantic_themes[idx]
            phrase_i = [phrase for phrase in phrase_i if phrase[1]>semantic_threshold]
            result_idx = {'semantics':phrase_i}

            if flag_entity:
                if isinstance(entity_themes[0], tuple):
                    entity_themes = [entity_themes]
                entity_i = entity_themes[idx]
                entity_i = [phrase for phrase in entity_i if phrase[1]>entity_threshold]
                result_idx['entity'] = entity_i

            if flag_summary:
                sum_text_i = sum_doc_df.iloc[idx]['summary_text']
                result_idx['summary'] = sum_text_i 

            if flag_doc:
                result_idx['document'] = docs[idx]
            output_ls.append(result_idx)
        
        return output_ls

    ## /Exposure function
    def generate_theme_exposure(self,
                                input_doc_dicts,
                                batch_size=None,
                                filter=False,
                                theme_level = None
                                ):
        """generate the thematic exposure of the input documents"""

        ######## Embedding Module ########
        # get the documents' composite embeddings
        embed_output_docs = self.embed_documents(input_doc_dicts, batch_size=batch_size)
        
        ######## Exposure Module ########
        # call the exposure generation function
        doc_exposue_df = self.backend_model.compute_theme_exposure(embeddings=embed_output_docs)

        if not filter:
            return doc_exposue_df
        # return the filtered output
        output_doc_dict = {}
        for idx in tqdm(range(len(input_doc_dicts))):
            doc_exposure_idx = doc_exposue_df.loc[:, idx].sort_values(ascending=False)
            res_exposure_idx = doc_exposure_idx[doc_exposure_idx>0]
            if len(res_exposure_idx)==0:   # case of no meaningful theme 
                output_doc_dict[idx] = None
                continue

            # reformat the level
            if theme_level:
                res_index = res_exposure_idx.index.get_level_values(theme_level)
                res_exposure_idx.index = res_index

            output_doc_dict[idx] = res_exposure_idx
        return output_doc_dict

    ## /Fitting function
    def fit(self, documents,
            embeddings=None,
            umap_embeddings=None, y=None):
        """fit the backend topic engine"""
        # fit the topic engine
        self.backend_model.fit(documents=documents,
                               embeddings=embeddings,
                               umap_embeddings=umap_embeddings,
                               y=y)
    
    def transform(self, documents):
        return

                            
    ## /Embedding functions 
    def embed_documents(self, input_doc_dicts, batch_size=None):
        """"""
        input_txt_ls = []
        embed_semantic_phrase_ls = []
        for idx in tqdm(range(len(input_doc_dicts))):
            input_theme_idx = input_doc_dicts[idx]
            # embed the phrases into weighted average
            embed_semantic_phrase = self.embed_phrases(input_theme_idx['semantics'])
            embed_semantic_phrase_ls.append(embed_semantic_phrase) 

            input_txt_ls.append(input_theme_idx['document'])
        embed_semantic_vec = np.stack(embed_semantic_phrase_ls)

        ## Step 2. embed summary
        embed_raw_doc_vec = self.embed_text_corpus(input_txt_ls, batch_size=batch_size)

        embed_output_docs = embed_raw_doc_vec*0.5 + embed_semantic_vec*0.5
        return embed_output_docs



    def embed_theme_corpus(self, input_theme_ls, batch_size=None):
        """wrap-up function to embed the list of formated theme input obj"""

        #### Step 1. embed phrases ####
        print(" => [Keyphrases] embedding process 💨")
        embed_ent_phrase_ls = []
        embed_semantic_phrase_ls = []
        for idx in tqdm(range(len(input_theme_ls))):
            input_theme_idx = input_theme_ls[idx]
            # embed the phrases into weighted average
            embed_entity_phrase = self.embed_phrases(input_theme_idx['entity'])
            embed_semantic_phrase = self.embed_phrases(input_theme_idx['semantics'])
            
            embed_ent_phrase_ls.append(embed_entity_phrase)
            embed_semantic_phrase_ls.append(embed_semantic_phrase) 

        # agg into final embeddings vector
        embed_ent_vec = np.stack(embed_ent_phrase_ls)
        embed_semantic_vec = np.stack(embed_semantic_phrase_ls)

        #### Step 2. embed summary ####
        print(" => [Summary] embedding process 💨")
        input_txt_ls = [y['summary'] for y in input_theme_ls]
        embed_summary_vec = self.embed_text_corpus(input_txt_ls, batch_size=batch_size)

        ## => final theme vector representation: a composite of phrase & summary
        embed_theme_vec = 0.3*embed_ent_vec+0.2*embed_semantic_vec+0.5*embed_summary_vec

        output_res_dict = {'entity':embed_ent_vec,
                           'semantic':embed_semantic_vec,
                           'summary':embed_summary_vec,
                           'theme':embed_theme_vec}
        
        return output_res_dict 

    def embed_phrases(self, input_phrase_ls):
        """embed the list of tuple with (phrase, relevance score)"""
        raw_phrase_ls = [x[0] for x in input_phrase_ls]
        raw_weight_ls = [x[1] for x in input_phrase_ls]
        # embed into vector space
        raw_phrase_vec = self.model.embed(raw_phrase_ls)

        if len(raw_phrase_vec)>1:
            embed_phrase_vec =  np.average(raw_phrase_vec,
                                           axis=0,
                                           weights=raw_weight_ls)
        elif len(raw_phrase_vec)==0: # corner case of empty phrases
            # use '' as a placeholder
            embed_phrase_vec = self.model.embed([''])[0]
        else:
            embed_phrase_vec = raw_phrase_vec[0]
        
        return embed_phrase_vec

    def embed_text_corpus(self, input_txt_ls, batch_size=None):
        """embed an list of text corpus into latent vector

        Arguments:
            batch_size: number of batch to preprocess the text in parallel for effeiciency 

        RETURN: a 2-d representation vector matrix in the shape of (num_inputs X num_latent_fac)
        """
        if batch_size is None:
            batch_size = 1
        
        output_embed_ls = []
        for idx in tqdm(range(0, len(input_txt_ls), batch_size)):
            batch_txt_ls = input_txt_ls[idx:(idx+batch_size)]
            
            embed_content = self.model.embed(batch_txt_ls)
            
            output_embed_ls.extend(embed_content)
        output_embed_vec = np.stack(output_embed_ls)

        return output_embed_vec 

    # core keyphrase extraction function
    def batch_extract_keywords(self,
                               docs: Union[str, List[str]],
                               candidates: List[str] = None,
                               keyphrase_ngram_range: Tuple[int, int] = (1, 3),
                               batch_size: int = 128, 
                               stop_words: Union[str, List[str]] = "english",
                               top_n: int = 5,
                               min_df: int = 1,
                               use_maxsum: bool = False,
                               use_mmr: bool = False,
                               diversity: float = 0.5,
                               nr_candidates: int = 20,
                               vectorizer: CountVectorizer = None,
                               highlight: bool = False,
                               seed_keywords: List[str] = None,
                               show_progress_bar: bool = False
        ):
        """Batch-level efficient keyphrase extraction 

        Input Args:
            batch_size: number of batch for processing

        RETURN: a list of the keyphrases pairs
        """
        output_doc_ls = []
        doc_len = len(docs)
        disable_progress_bar = not show_progress_bar

        for start_idx in tqdm(range(0, doc_len, batch_size), disable=disable_progress_bar):
            doc_batch_ls = docs[start_idx:(start_idx+batch_size)]

            # encoe the bacth-level doc
            doc_encoded_ls = self.extract_keywords(doc_batch_ls,
                                                   keyphrase_ngram_range=keyphrase_ngram_range,
                                                   top_n=top_n,
                                                   stop_words=stop_words,
                                                   candidates=candidates,
                                                   use_maxsum=use_maxsum,
                                                   use_mmr=use_mmr,
                                                   diversity=diversity,
                                                   seed_keywords=seed_keywords,
                                                   highlight=False,
                                                   show_progress_bar=False
                                                   )

            output_doc_ls.extend(doc_encoded_ls)

        return output_doc_ls




    def extract_keywords(
        self,
        docs: Union[str, List[str]],
        candidates: List[str] = None,
        keyphrase_ngram_range: Tuple[int, int] = (1, 3),
        stop_words: Union[str, List[str]] = "english",
        top_n: int = 5,
        min_df: int = 1,
        use_maxsum: bool = False,
        use_mmr: bool = False,
        diversity: float = 0.5,
        nr_candidates: int = 20,
        vectorizer: CountVectorizer = None,
        highlight: bool = False,
        seed_keywords: List[str] = None,
        show_progress_bar: bool = False
    ) -> Union[List[Tuple[str, float]], List[List[Tuple[str, float]]]]:
        """Extract keyphrases

        Arguments:
            docs: The document(s) for which to extract keywords/keyphrases
            candidates: Candidate keywords/keyphrases to use instead of extracting them from the document(s)
                        NOTE: This is not used if you passed a `vectorizer`.
            keyphrase_ngram_range: Length, in words, of the extracted keywords/keyphrases.
                                   NOTE: This is not used if you passed a `vectorizer`.
            stop_words: Stopwords to remove from the document.
                        NOTE: This is not used if you passed a `vectorizer`.
            top_n: Return the top n keywords/keyphrases
            min_df: Minimum document frequency of a word across all documents
                    if keywords for multiple documents need to be extracted.
                    NOTE: This is not used if you passed a `vectorizer`.
            use_maxsum: Whether to use Max Sum Distance for the selection
                        of keywords/keyphrases.
            use_mmr: Whether to use Maximal Marginal Relevance (MMR) for the
                     selection of keywords/keyphrases.
            diversity: The diversity of the results between 0 and 1 if `use_mmr`
                       is set to True.
            nr_candidates: The number of candidates to consider if `use_maxsum` is
                           set to True.
            vectorizer: Pass in your own `CountVectorizer` from
                        `sklearn.feature_extraction.text.CountVectorizer`
            highlight: Whether to print the document and highlight its keywords/keyphrases.
                       NOTE: This does not work if multiple documents are passed.
            seed_keywords: Seed keywords that may guide the extraction of keywords by
                           steering the similarities towards the seeded keywords.

        Returns:
            keywords: The top n keywords for a document with their respective distances
                      to the input document.

        Usage:

        To extract keyphrases from a single document
        """
        ########     Validation     ########
        # Check for a single, empty document
        if isinstance(docs, str):
            if docs:
                docs = [docs]
            else:
                return []

        #################################################################
        ########                 Vectorization                    #######
        # Extract potential words using a vectorizer / tokenizer
        if isinstance(vectorizer, str):
            if vectorizer=='entity':   # set up the entity vectorizer
                vectorizer = self.entity_vectorizer

        if vectorizer:
            count = vectorizer.fit(docs)
            if count is None:   # case of empty Keyphrases
                return []
        else:
            try:
                count = CountVectorizer(
                    ngram_range=keyphrase_ngram_range,
                    stop_words=stop_words,
                    min_df=min_df,
                    vocabulary=candidates,
                ).fit(docs)
            except ValueError:
                return []

        # Scikit-Learn Deprecation: get_feature_names is deprecated in 1.0
        # and will be removed in 1.2. Please use get_feature_names_out instead.
        if version.parse(sklearn_version) >= version.parse("1.0.0"):
            words = count.get_feature_names_out()
        else:
            words = count.get_feature_names()
        df = count.transform(docs)

        #################################################################
        ########               Extract embeddings                 #######
        doc_embeddings = self.model.embed(docs)
        word_embeddings = self.model.embed(words)

        # Find keywords
        all_keywords = []
        disable_progress_bar = not show_progress_bar
        for index, _ in tqdm(enumerate(docs), disable=disable_progress_bar):

            try:
                # Select embeddings
                candidate_indices = df[index].nonzero()[1]
                candidates = [words[index] for index in candidate_indices]
                candidate_embeddings = word_embeddings[candidate_indices]
                doc_embedding = doc_embeddings[index].reshape(1, -1)

                ## 💡 Guided ThemeBERT with seed keywords
                if seed_keywords is not None:
                    seed_embeddings = self.model.embed([" ".join(seed_keywords)])
                    doc_embedding = np.average(
                        [doc_embedding, seed_embeddings], axis=0, weights=[3, 1]
                    )

                # Maximal Marginal Relevance (MMR)
                if use_mmr:
                    keywords = mmr(
                        doc_embedding,
                        candidate_embeddings,
                        candidates,
                        top_n,
                        diversity,
                    )

                # Max Sum Distance
                elif use_maxsum:
                    keywords = max_sum_distance(
                        doc_embedding,
                        candidate_embeddings,
                        candidates,
                        top_n,
                        nr_candidates,
                    )

                # Cosine-based keyword extraction
                else:
                    distances = cosine_similarity(doc_embedding, candidate_embeddings)
                    keywords = [
                        (candidates[index], round(float(distances[0][index]), 4))
                        for index in distances.argsort()[0][-top_n:]
                    ][::-1]

                all_keywords.append(keywords)

            # Capturing empty keywords
            except ValueError:
                all_keywords.append([])

        # Highlight keywords in the document
        if len(all_keywords) == 1:
            if highlight:
                highlight_document(docs[0], all_keywords[0], count)
            all_keywords = all_keywords[0]

        return all_keywords

    ########## Foundational functionality ##########
    def write_local_log(self, file_name):
        """locally keep log of the inited model"""
        self.logger.write_local_log(file_name)

    def _load_summarizer(self, summary_model='BARTnews-summarizer'):
        self.summarizer = TransformerSummarizer(device=self.device,
                                                model=summary_model,
                                                flag_hard_max_length=self.hard_max_length,
                                                max_length=self.sum_max_length)
