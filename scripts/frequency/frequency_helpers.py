"""
Helper method for computing ngram frequency.
"""
from sklearn.feature_extraction.text import CountVectorizer
from data_handler import get_default_stopwords
import pandas as pd
import numpy as np

def get_ngram_frequency(text_iter, 
                        stopwords=None, min_df=5,
                        ngram_range=(1,3),
                        vocab=None):
    """
    Compute raw ngram frequency over documents.

    Parameters:
    ----------
    text_iter : generator
    Document generator.
    stopwords : [str]
    If None, use default words.
    min_df : int
    Minimum document frequency to be counted.
    ngram_range : (int, int)
    Min and max ngrams to compute.
    vocab : [str]
    Optional vocabulary to restrict the search space.
    
    Returns:
    --------
    ngram_frequency : pandas.DataFrame
    Ngram and count.
    """
    if(stopwords is None):
        stopwords = get_default_stopwords()
    if(vocab is not None):
        cv = CountVectorizer(stop_words=stopwords, min_df=min_df,
                             ngram_range=ngram_range, vocabulary=vocab)
    else:
        cv = CountVectorizer(stop_words=stopwords, min_df=min_df,
                             ngram_range=ngram_range)
    counts = cv.fit_transform(text_iter)
    print('got counts %s'%(counts))
    counts = list(np.array(counts.sum(axis=0))[0])
    sorted_vocab = sorted(cv.vocabulary_.keys(),
                          key=lambda v: cv.vocabulary_[v])
    ngram_frequency = dict(zip(sorted_vocab, counts))
    ngram_frequency = pd.DataFrame(dict(ngram_frequency), index=['count']).transpose()
    return ngram_frequency

