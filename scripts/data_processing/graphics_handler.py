"""
Methods to help plot time series,
"""
from __future__ import division
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def plot_word_niche_over_time(test_word, embeddings, token_counts, top_n=5, stacked=False):
    """
    Plot the frequency of a word and the frequency of its 
    top-n most similar words over time.

    parameters:
    -----------
    test_word = str
    embeddings = gensim.models.Word2Vec
    token_counts = pandas.DataFrame
    # rows = words, columns = timesteps
    top_n = int
    stacked = bool
    # whether to stack the time series as an area line plot
    """
    similar_words, sim_scores = zip(*embeddings.most_similar(test_word, topn=top_n))
    similar_words = list(map(str, similar_words))
    all_words = [test_word] + similar_words
    sorted_dates = sorted(token_counts.columns)
    # assume that dates are year-month format!
    timesteps = [datetime.strptime(d, '%Y-%m') for d in sorted_dates]
    plt.figure(figsize=(10,10))
    cmap = plt.cm.get_cmap('Accent')
    if(stacked):
        separate_series = token_counts.loc[all_words]
        separate_series = separate_series.values
        colors = [cmap(i / separate_series.shape[0]) for i in range(len(all_words))]
        plt.stackplot(timesteps, separate_series, labels=all_words, colors=colors)
    else:
        for i, w in enumerate(all_words):
            color = cmap(i / len(all_words))
            plt.plot(timesteps, token_counts.loc[w], 
                     label=w,
                     color=color)
    plt.legend(loc='upper right')
    plt.show()
