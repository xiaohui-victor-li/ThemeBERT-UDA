# Topic Evaluation Functionality
# --------------------------------
# - latest @11/21/2022; inception @11/20/2022
#
# 
# -> Topic Coherence
#       Jey Han Lau, David Newman, and Timothy Baldwin. 2014. Machine reading tea leaves: Automatically evaluating topic coherence and topic model quality.
# -> Topic Diversity:
#       Adji B Dieng, Francisco JR Ruiz, and David M Blei. 2020. Topic modeling in embedding spaces.
#
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from typing import List, Tuple, Union, Mapping, Any


# ✨Core Function: Topic Coherence 
def compute_topic_TC(topic_word_ls: List[str],
                     word_verse_counter: CountVectorizer,
                     word_freq_mat: np.ndarray):
    """compute the TC for specific topic given the top words
    
    Arguments:
    ---------
    :param topic_word_ls: a list of representative words given the topic
    :param word_verse_counter: the fitted CountVectorizer object storing the word frequency
    :param word_freq_mat: 2d numpy sparse boolean array of shape (num_doc, num_word)
    """
    word_verse_keys = word_verse_counter.vocabulary_.keys()
    topic_filter_word_ls= [word_verse_counter.vocabulary_[word] for word in topic_word_ls if word in word_verse_keys]
    
    if len(topic_filter_word_ls)==0:   # no filter word exist => all stop words
        # return NaN
        return np.NaN
    
    ##### calculate the pairwise NPMI #####
    topic_NPMI_ls = []
    for word_idx in topic_filter_word_ls:
        for word_idy in topic_filter_word_ls:
            if word_idx>=word_idy: 
                continue
            # calculate the NPMI
            word_NPMI = compute_NPMI(word_idx, word_idy, word_freq_mat)
            topic_NPMI_ls.append(word_NPMI)
            
    return np.mean(topic_NPMI_ls)
    
# building-block pairwise computation function
def compute_word_count(word_idx_i, word_freq_mat, word_idx_j=None):
    """calculate the normalized pointwise mutual information"""
    # boolean matrix of the word occurence
    word_bool_vec_i = word_freq_mat[:, word_idx_i].toarray()
    
    if word_idx_j is None:
        # Case: Marginal Prob for single word i 
        word_count = word_bool_vec_i.mean()
    else:
        # Case: join prob for pair word i & j
        word_bool_vec_j = word_freq_mat[:, word_idx_j].toarray()
        word_bool_mutual = word_bool_vec_i & word_bool_vec_j
        word_count = word_bool_mutual.mean()
    
    return word_count

def compute_NPMI(word_idx_i, word_idx_j, word_freq_mat):
    """calculate the pairwise NPMI among word pair (idx_i, idx_j)"""
    marginal_prob_i = compute_word_count(word_idx_i, word_freq_mat)
    marginal_prob_j = compute_word_count(word_idx_j, word_freq_mat)
    # join prob
    join_prob_ij = compute_word_count(word_idx_i, word_freq_mat,
                                      word_idx_j=word_idx_j)
    if join_prob_ij<=0:   # independent
        return 0 
    
    PMI = np.log(join_prob_ij)-np.log(marginal_prob_i)-np.log(marginal_prob_j)
    norm_PMI = PMI/(-np.log(join_prob_ij))
    
    return norm_PMI
    
# ✨Core Function: Topic Diversity
def compute_topic_diversity(topic_word_ls):
    """calculate Topic Diversity"""
    num_topics = len(topic_word_ls)
    
    word_ls =[]
    for k in range(num_topics):
        word_ls.extend(topic_word_ls[k])
    n_unique = len(set(word_ls))
    TD = n_unique / len(word_ls)
    
    return TD
