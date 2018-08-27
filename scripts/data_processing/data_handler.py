"""
Methods to handle  data in a variety
of contexts.
"""
from bz2 import BZ2File
import re, json, os
from nltk.tokenize.casual import TweetTokenizer
from stopwords import get_stopwords
import pandas as pd
import numpy as np
from numpy import piecewise
from gensim.models import Word2Vec
from datetime import datetime
from itertools import izip
from math import floor, ceil
from sklearn.preprocessing import MinMaxScaler

# minimum document/total frequency for inclusion
MIN_COUNT=1
MAX_VOCAB_SIZE=100000
SUB_REGEX = re.compile(r'r/\w+')
USER_REGEX = re.compile(r'u/\w+')
PUNCT=['.', ',', '!', '?', ':', ';', 
       '(', ')', '[', ']', '"', '\'']
def get_dummy_embeddings(vocab, min_count=5, dims=100, window_size=5):
    """
    Generate dummy embeddings from
    basic vocabulary to later train/update.

    parameters:
    ---------
    vocab = [str]
    min_count = int
    # minimum frequency for word to be counted in model
    dims = int
    # embedding dimensions
    window_size = int
    # window size for embeddings
    returns:
    --------- 
    model = Word2Vec
    """
    model = Word2Vec(size=dims, window=window_size, min_count=min_count)
    model.build_vocab(sentences=([str(v)]*min_count for v in vocab))
    return model

def get_mean_df(dataframes):
    """
    Compute mean values of all cells 
    across dataframes. Replace NAN with 0.
    Take no prisoners!!
    
    Parameters:
    -----------
    dataframes : [pandas.DataFrame]
    
    Returns:
    --------
    dataframe_mean : pandas.DataFrame
    """
    N = len(dataframes)
    vocab = reduce(lambda x,y: x|y, [set(d.index) for d in dataframes])
    dataframes = [d.loc[vocab].fillna(0, inplace=False) for d in dataframes]
    dataframe_sum = reduce(lambda x,y: x+y, dataframes)
    dataframe_mean = dataframe_sum / N
    return dataframe_mean

def build_embeddings_from_tf(word_tf, stopwords, window_size, cutoff):
    """
    Build word embeddings using vocab from 
    word tf matrix and including stopwords.
    ONLY INCLUDE words with mean frequency above
    specified cutoff.

    parameters:
    -----------
    word_tf = pandas.DataFrame
    # rows = words, columns = time
    stopwords = [str]
    # need to include stopwords in vocab
    window_size = int
    cutoff = int
    # minimum average count across all time 
    # in word_tf necessary to be included in vocab

    returns:
    --------
    model = Word2Vec
    """
    word_tf_means = word_tf.mean(axis=1)
    vocab_cutoff = word_tf.apply(lambda r : r.mean() > cutoff, axis=1)
    vocab_cutoff = vocab_cutoff[vocab_cutoff].index
    vocab = vocab_cutoff.index.tolist() + stopwords
    model = get_dummy_embeddings(vocab, dims=dims,
                                 window_size=window_size, 
                                 min_count=vocab_cutoff)
    return model

def get_default_vocab(vocab_file='../../data/frequency/2013_2016_top_100000_vocab.tsv',
                      frequency_file='../../data/frequency/2013_2016_tf_norm.tsv',
                      top_k=100000):
    """
    Load default top-k vocab, or build
    file if it doesn't exist.
    
    Parameters:
    -----------
    vocab_file : str
    Total frequency indexed by word.
    frequency_file : str
    Monthly frequency indexed by word.
    top_k : int
    Top k words to return.
    
    Returns:
    --------
    vocab : [str]
    """
    if(not os.path.exists(vocab_file)):
        frequency = pd.read_csv(frequency_file, sep='\t', index_col=0)
        frequency = frequency.sum(axis=1)
        frequency.sort_values(inplace=True, ascending=False)
        try:
            frequency.dropna(inplace=True, axis=0)
        except Exception, e:
            pass
        # remove bogus words
        nonalphas = [i for i in frequency.index if not str(i).isalpha()]
        frequency.to_csv(vocab_file, sep='\t')
        vocab_frequencies = frequency[:top_k]
    else:
        vocab_frequencies = pd.read_csv(vocab_file, sep='\t', index_col=0)[:top_k]
    vocab = vocab_frequencies.index.tolist()
    return vocab

def collect_all_vocab(comment_files, tokenizer,
                      stopwords):
    """
    Collect the set of all vocabulary 
    in comment files.

    parameters:
    -----------
    comment_files = [str]
    tokenizer = nltk.Tokenizer
    stopwords = [str]

    returns:
    --------
    vocab_set = {str}
    """
    vocab_set = set()
    for comment_file in comment_files:
        print('processing vocab in file %s'%(comment_file))
        comment_iter = CommentIter(comment_file,
                                   tokenizer, stopwords, 
                                   return_full_comment=False)
        for c in comment_iter:
            v = set(c)
            vocab_set.update(v)
    return vocab_set

def get_default_tokenizer():
    return TweetTokenizer(preserve_case=False)

def get_default_stopwords():
    return get_stopwords('en')

def get_default_spammers(spammer_file='../../data/metadata/spammers.txt'):
    """
    Get default list of Reddit users known
    to be spammers (identified by Tan and Lee (2015)).

    parameters:
    -----------
    spammer_file = str

    returns:
    --------
    spammers = []
    """
    spammers = [l.strip().lower() for l in open(spammer_file, 'r')]
    return spammers

def get_default_bots(bot_file='../../data/metadata/bots.txt'):
    """
    Get default list of Reddit users known
    to be bots (identified by Tan and Lee (2015)).

    parameters:
    -----------
    bot_file = str

    returns:
    --------
    bots = []
    """
    bots = [l.strip().lower() for l in open(bot_file, 'r')]
    return bots

def get_default_communities(community_counts_file='../../data/community_stats/2015_2016_community_counts.tsv',
                            start_index=10, k=20):
    """
    Get default relevant communities by taking the 
    overall community counts, sorting by descending frequency,
    then getting the top [start_index:start_index+k] 
    communities (not too big, not too small).

    parameters:
    -----------
    community_counts_file = str
    # .tsv with community and count columns
    start_index = int
    k = int
    
    returns:
    --------
    communities = [str]
    """
    community_counts = pd.read_csv(community_counts_file, sep='\t',
                                   index_col=0)
    community_counts.sort('count', ascending=False, inplace=True)
    communities = list(map(str.lower, community_counts.index[start_index:start_index+k].tolist()))
    return communities

def get_community_languages(community_language_file='../../data/community_stats/community_lang_sample.tsv'):
    """
    Get language distribution across top 500 communities, 
    based on a sample of data from one month of comments.
    
    Parameters:
    -----------
    community_language_file : str
    Rows = languages, columns = communities.
    
    Returns:
    --------
    lang_distribution : pandas.DataFrame
    Rows = languages, columns = communities.
    """
    lang_distribution = pd.read_csv(community_language_file, sep='\t', index_col=0)
    return lang_distribution

def get_non_english_communities(non_english_file='../../data/community_stats/non_english_communities.txt'):
    """
    Get list of communities (from top 500 list)
    with less than 80% of posts in English.
    This was manually curated.

    Returns:
    non_english_communities : [str]
    Lower-cased community names.
    """
    non_english_communities = [l.lower().strip() for l in 
                               open(non_english_file, 'r')]
    return non_english_communities

def get_active_users(user_comment_count_file='../../data/users/user_comment_counts.tsv',
                     active_user_cutoff=100):
    """
    Get active users from user comment counts, 
    extracting all users over a given cutoff.

    parameters:
    -----------
    user_comment_count_file = str
    active_user_cutoff = int

    returns:
    --------
    active_users = [str]
    """
    user_comment_counts = pd.read_csv(user_comment_count_file, 
                                      sep='\t', index_col=0)
    user_comment_counts.sort('count', inplace=True, ascending=False)
    active_users = user_comment_counts.index[:active_user_cutoff].tolist()
    return active_users

def clean_stats(stats, vocab):
    """
    Insert empty values for missing vocabulary
    in stats series and smooth for 0 values
    (smooth = .1 times minimum value).

    Parameters:
    -----------
    stats : pandas.Series
    vocab : [str]
    
    Returns:
    --------
    clean : pandas.Series
    """
    clean = stats.loc[vocab]
    clean.fillna(0, inplace=True)
    smooth = 1e-1 * clean[clean > 0].min()
    clean += smooth
    return clean

def smooth_stats(stats):
    """
    Smooth stats by adding non-zero minimum.
    This only works for non-negative stats!

    Parameters:
    -----------
    stats : pandas.DataFrame
    
    Returns:
    --------
    smoothed : pandas.DataFrame
    """
    smoothr = stats[stats > 0.].min().min()
    smoothed = stats + smoothr
    return smoothed

def extract_year_month(file_name):
    """
    Extract the year and month from a file containing
    the date in YEAR-MM format (e.g. 2014-09 = Sept 2014).
    
    parameters:
    -----------
    file_name = str

    returns:
    --------
    year = int
    month = int
    """
    extracted = re.findall(r'201[0-9]-[0-9]{2}', file_name)[0]
    year, month = list(map(int, extracted.split('-')))
    return year, month

def get_all_comment_files(data_dir='/hg190/corpora/reddit_comment_data/monthly_submission/',
                          # data_dir='/mnt/new_hg190/corpora/reddit_comment_data/monthly_submission/',
                          years=['2015', '2016'],
                          year_month_combos=['2015-06', '2015-07', '2015-08', '2015-09', '2015-10', '2015-11',
                                             '2015-12', '2016-01','2016-02', '2016-03', '2016-04', '2016-05']
                          ):
    """
    Collect all zipped monthly comment files from yearly data directories.
    
    parameters:
    -----------
    data_dir = str
    years = [str]
    year_month_combos = [str]

    returns:
    --------
    comment_files = [str]
    """
    all_comment_files = []
    for y in years:
        full_data_dir = os.path.join(data_dir, y)
        comment_files = sorted([f for f in os.listdir(full_data_dir)
                                if re.findall('RC_201[0-9]-[0-9]{2}.bz2', f)])
        # filter for months
        if(year_month_combos is not None):
            comment_files = list(filter(lambda x: re.findall('201[0-9]-[0-9]{2}', x)[0] in year_month_combos,
                                        comment_files))
        comment_files = [os.path.join(full_data_dir, c)
                         for c in comment_files]
        all_comment_files += comment_files
    return all_comment_files

def clean_text(text):
    """
    Remove Reddit-specific traits from
    text (e.g. r/subname and u/username).
    
    parameters:
    -----------
    text = str
    
    returns:
    --------
    text = str
    """
    text = re.sub(SUB_REGEX, '', text)
    text = re.sub(USER_REGEX, '', text)
    return text

class CommentIter(object):
    """
    Comment iterator to automatically
    clean and return text from comment file.
    """
    def __init__(self, comment_file, 
                 tokenizer=None, 
                 stopwords=None, 
                 debug=True,
                 return_full_comment=False,
                 filter_community=None,
                 cutoff=np.inf):
        """
        Create comment iterator.
        
        parameters:
        ----------
        comment_file = str
        tokenizer = nltk.tokenize.Tokenizer
        stopwords = {str}
        debug = bool
        return_raw_comment = bool
        # return raw comment JSON instead
        # of filtered/tokenized text
        filter_community = str
        cutoff = int
        # total number of comments to output
        """
        self.comments = BZ2File(comment_file, 'r')
        if(tokenizer is None):
            tokenizer = get_default_tokenizer()
        self.tokenizer = tokenizer
        if(stopwords is None):
            stopwords = get_default_stopwords()
        self.stopwords = stopwords
        self.debug = debug
        self.ctr = 0
        self.return_full_comment = return_full_comment
        self.filter_community = filter_community
        self.cutoff = cutoff

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()
    
    def process_text(self, text):
        text = clean_text(text)
        tokenized_text = self.tokenizer.tokenize(text)
        filtered_text = list(filter(lambda x: x not in self.stopwords and x.isalpha(),
                                    tokenized_text))
        return filtered_text

    def next(self):
        """
        Reads next comment and returns
        sliding windows over text.
        
        returns:
        --------
        comment or filtered_text: dict or [str]
        """
        try:
            l = self.comments.next()
            comment = json.loads(l)
            # TODO: catch for broken comments?
            # if community doesn't match, return empty string
            if(self.filter_community and 
               comment['subreddit'].lower() != self.filter_community):
                comment = ['']
            # otherwise return full comment
            # or just the text
            else:
                text = comment['body']
                text = self.process_text(text)
                comment['body'] = text
                if(not self.return_full_comment):
                    comment = text
                self.ctr += 1
                if(self.ctr % 100000 == 0):
                    print('processed %d comments'%(self.ctr))
                if(self.ctr > self.cutoff):
                    self.end()
            return comment
        # if file ends abruptly then we need to stop iteration
        except Exception as e:
            self.end()

    def end(self):
        self.comments.close()
        raise StopIteration

class TextWindowIter(object):
    """
    Iterates through text to generate sliding windows
    for applications like SVD word embeddings.
    """

    def __init__(self, text_file, 
                 window_size,
                 tokenizer=None,
                 stopwords=None,
                 cutoff=np.inf):
        self.text = BZ2File(text_file, 'r')
        if(tokenizer is None):
            tokenizer = get_default_tokenizer()
        if(stopwords is None):
            stopwords = get_default_stopwords()
        self.tokenizer = tokenizer
        self.stopwords = stopwords
        self.window_size = window_size
        self.padding_size = int(window_size / 2)
        self.ctr = 0
        self.window_ctr = 0
        self.cutoff = cutoff

    def __iter__(self):
        return self
        
    def __next__(self):
        return self.next()

    def next(self):
        """
        Generate *list* of windows, to 
        be decomposed downstream.
        """
        try:
            l = self.text.next()
            tokenized = self.process_text(l)
            windows = []
            for i in range(len(tokenized) - self.window_size + 1):
                window = tokenized[i:i+self.window_size]
                windows.append(window)
                self.window_ctr += 1
            self.ctr += 1
            if(self.ctr % 100000 == 0):
                print('processed %d comments and %d windows'%
                      (self.ctr, self.window_ctr))
            if(self.ctr > self.cutoff):
                self.end()
            return windows
        # if file ends abruptly then we need to stop iteration
        except Exception as e:
            # print('stopped because exception %s'%(e))
            self.end()

    def end(self):
        self.text.close()
        raise StopIteration
        
    def process_text(self, text):
        """
        Tokenize, filter and pad text.
        """
        text = clean_text(text)
        tokenized_text = self.tokenizer.tokenize(text)
        filtered_text = list(filter(lambda x: x not in self.stopwords and x.isalpha(),
                                    tokenized_text))
        # TODO: how much to pad start and end?? more padding will help short documents I think??
        processed_text = ((['START']*(self.padding_size) + 
                           filtered_text + 
                           ['END']*(self.padding_size)))
        return processed_text

class DateIter(object):
    def __init__(self, comment_iter,
                 start_date, end_date):
        self.comments = comment_iter
        self.start_date = start_date
        self.end_date = end_date
    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        """
        Only return comments
        that fall in date range.
        """
        try:
            comment = self.comments.next()
            comment_utc = int(comment['created_utc'])
            comment_date = datetime.fromtimestamp(comment_utc)
            if(comment_date >= self.start_date and comment_date < self.end_date):
                txt = comment['body']
                return ' '.join(txt)
            else:
                return ''
        except Exception, e:
            print('stop because exception %s'%(e))
            self.comments.end()

def combine_data_files(combined_name, data_dir, 
                       file_matcher, date_matcher, 
                       out_dir):
    """
    Read and combine data files (each having a date column and word indices), 
    then write to data frame with combined name.

    Parameters:
    -----------
    combined_name : str
    data_dir : str
    file_matcher : re.Pattern
    Regular expression to match data file names.
    date_matcher : re.Pattern
    Regular expression to extract date string from file name.
    out_dir : str
    
    Returns:
    --------
    full_data : pandas.DataFrame
    rows = words, cols = dates
    """
    data_files = sorted(
        [os.path.join(data_dir, f)
         for f in os.listdir(data_dir)
         if len(file_matcher.findall(f)) > 0]
    )
    print('got data files %s'%(str(data_files)))
    # collect data
    data_list = []
    for f in data_files:
        data_time = date_matcher.findall(f)[0]
        print('reading file %s with time %s'%
              (f, data_time))
        data = pd.read_csv(f, sep='\t', index_col=0)
        data.columns = [data_time]
        # drop duplicate indices??
        index = data.index.drop_duplicates()
        data = data.loc[index]
        print('got data with shape %s'%(str(data.shape)))
        data_list.append(data)
    full_data = reduce(lambda x,y: x.join(y),
                       data_list)
    # for some reason concat doesn't work but join does
    # full_data = pd.concat(data_list, axis=1)
    full_data.fillna(0, inplace=True)
    full_data = full_data[sorted(full_data.columns)]
    out_file = os.path.join(out_dir, 
                            '%s.tsv'%(combined_name))
    full_data.to_csv(out_file, sep='\t')
    return full_data

def pairwise(iterable):
    a = iter(iterable)
    return izip(a, a)

def to_series(x):
    """
    Convert string to Series object.

    Parameters:
    -----------
    x : str
    
    Returns:
    --------
    series = pandas.Series
    """
    return pd.Series(dict([(x, float(y)) for x,y in pairwise(re.split('\s+', x)[:-2])]))

def melt_frame(df, value_name):
    """
    Convert DataFrame to word|date|value format.
    
    Parameters:
    -----------
    df : pandas.DataFrame
    Rows = words, cols = dates.
    value_name : str
    
    Returns:
    --------
    df_melt : pandas.DataFrame
    Rows = samples, cols = word, date, value.
    """
    df_melt = df.copy()
    df_melt['word'] = df_melt.index
    df_melt = pd.melt(df_melt, id_vars=['word'], var_name='date', value_name=value_name)
    return df_melt

def melt_frames(df_list, value_names):
    """
    Convert list of DataFrames to word|date|value format,
    then merge on word and date to make one 
    giant frame.
    
    Parameters:
    -----------
    df_list : [pandas.DataFrame]
    Rows = words, cols = dates.
    value_names : [str]
    
    Returns:
    --------
    df_combined : pandas.DataFrame
    """
    N = len(df_list)
    df_melt_list = [melt_frame(df_list[i], value_names[i]) for i in range(N)]
    df_combined = reduce(lambda l,r: pd.merge(l,r,on=['word','date'], how='inner'), df_melt_list)
    return df_combined

def get_bin_probs(x, y, bins=5):
    """
    Bin the x values into equal-sized bins, then
    compute the probability of y == 1 in each bin.
    ASSUMES THAT X AND Y HAVE SAME INDEX.
    
    Parameters:
    -----------
    x : pandas.Series
    y : pandas.Series
    Binary series, i.e. 0 or 1.
    bins : int
    
    Returns:
    --------
    bin_probs : pandas.DataFrame
    Rows = samples, cols = lower, upper, y_prob.
    """
    _, x_edges = np.histogram(x, bins=bins)
    x_edges = zip(x_edges[:-1], x_edges[1:])
    bin_probs = []
    for x_lower, x_upper in x_edges:
        prob = y[(x >= x_lower) & (x < x_upper)].mean()
        bin_probs.append(prob)
    bin_probs = pd.DataFrame({'lower' : zip(*x_edges)[0], 'upper' : zip(*x_edges)[1], 'prob' : bin_probs})
    return bin_probs

def get_binned_stats(x_stat, y_stat, x_bin_width = 0.25, lower_percentile=10, upper_percentile=90):
    """
    Compute lower percentile, median and upper percentile for all y_stat values
    within each specified x_stat bin.
    
    Parameters:
    -----------
    x_stat : pandas.Series
    y_stat : pandas.Series
    x_bin_width : float
    lower_percentile : int
    upper_percentile : int
    
    Returns:
    --------
    x_vals : pandas.Series
    y_vals : pandas.DataFrame
    lower_percentile_vals : pandas.Series
    median_vals : pandas.Series
    upper_percentile_vals : pandas.Series
    """
    lower = floor(x_stat.min())
    upper = ceil(x_stat.max())
    x_bin_width = 0.25
    x_count = 1 / x_bin_width
    num = (upper - lower) * x_count + 1
    x_bins = np.linspace(lower, upper, num=num)
    x_bins = zip(x_bins, x_bins + x_bin_width)
    lowers = zip(*x_bins)[0]
    x_vals = []
    lower_percentile_vals = []
    upper_percentile_vals = []
    median_vals = []
    for lower, upper in x_bins:
        relevant_idx = x_stat[(x_stat >= lower) & (x_stat < upper)].index
        if(len(relevant_idx) > 0):
            y_stat_relevant = y_stat.loc[relevant_idx]
            # get lower percentile
            lower_percentile_val = np.percentile(y_stat_relevant, lower_percentile)
            lower_percentile_vals.append(lower_percentile_val)
            # get upper percentile
            upper_percentile_val = np.percentile(y_stat_relevant, upper_percentile)
            upper_percentile_vals.append(upper_percentile_val)
            # get median
            median = np.median(y_stat_relevant)
            median_vals.append(median)
            # get lower bound as x value
            x_vals.append(lower)
    x_vals = pd.Series(x_vals)
    lower_percentile_vals = pd.Series(lower_percentile_vals)
    median_vals = pd.Series(median_vals)
    upper_percentile_vals = pd.Series(upper_percentile_vals)
    return x_vals, lower_percentile_vals, median_vals, upper_percentile_vals

def compare_stat_diff(stat, stat_name, group_1, group_2, group_1_name, group_2_name):
    """
    Compute difference between group_1 and 
    group_2 word pairs, then combine 
    both the raw stat and the difference 
    into one dataframe.

    Parameters:
    -----------
    stat : pandas.Series
    stat_name : str
    group_1 : [str]
    group_2 : [str]
    group_1_name : str
    group_2_name : str
    
    Returns:
    --------
    stat_diff : pandas.DataFrame
    Rows = samples, cols = diff|group_1|group_1_val|group_2|group_2_val
    """
    stat1 = stat.loc[group_1].values
    stat2 = stat.loc[group_2].values
    diff = stat1 - stat2
    diff_name = '%s_diff'%(stat_name)
    stat_diff = pd.DataFrame({group_1_name : group_1, group_2_name : group_2, 
                              '%s_%s'%(group_1_name, stat_name) : stat1,
                              '%s_%s'%(group_2_name, stat_name) : stat2, 
                              diff_name : diff})
    stat_diff.sort_values(diff_name, inplace=True, ascending=True)
    return stat_diff

def piecewise_linear(x, x0, y0, k1, k2):
    """
    Compute the fit piecewise function
    given the function dependent variable
    and parameters.
    
    Parameters:
    -----------
    x : array-like
    x0 : float
    x coordinate of piecewise break point.
    y0 : float
    y coordinate of piecewise break point.
    k1 : float
    Slope of first piece.
    k2 : float
    Slope of second piece.
    
    Returns:
    ---------
    y : numpy.array
    """
    piece_func_1 = lambda x: k1*x + y0-k1*x0
    piece_func_2 = lambda x: k2*x + y0-k2*x0
    y = piecewise(x, [x < x0], [piece_func_1, piece_func_2])
    return y

def get_delta(stat, delta_offset=1):
    """
    Compute delta between timesteps.
    
    Parameters:
    -----------
    stat : pandas.DataFrame
    Rows = words, columns = dates.
    delta_offset : int
    
    Returns:
    --------
    delta_stat : pandas.DataFrame
    Rows = words, columns = dates (not including first delta_offset dates).
    """
    delta_stat = pd.DataFrame(stat.ix[:, delta_offset:].values - stat.ix[:, :-delta_offset].values,
                              index=stat.index, columns=stat.columns[delta_offset:])
    return delta_stat

def get_split_point_stats(words, stat, split_points, k):
    """
    Get stats from k timesteps before split point 
    to k timesteps after split point.
    
    Parameters:
    -----------
    words : [str]
    stat : pandas.DataFrame
    split_points : pandas.Series
    k : int
    
    Returns:
    --------
    split_point_stats : pandas.DataFrame
    Rows = words, cols = split point - k : split point + k
    """
    N = stat.shape[1]
    split_point_stats = [stat.ix[[w], split_points[w] - k : split_points.loc[w] + k + 1].values
                                for w in words]
    cols = range(-k, k+1)
    split_point_stats = pd.DataFrame(pd.np.vstack(split_point_stats), index=words, columns=cols)
    return split_point_stats

def normalize_time_series(stats, scaler=None):
    """
    Normalize each time series in matrix
    to a fixed range (e.g. all time series in (0,1) range).
    
    Parameters:
    -----------
    stats : pandas.DataFrame
    scaler : transformer
    Default is sklearn.preprocessing.MinMaxScaler.

    Returns:
    --------
    stats_rescaled : pandas.DataFrame
    """
    if(scaler is None):
        scaler = MinMaxScaler(feature_range=(0,1))
    stats_rescaled = stats.apply(lambda y: scaler.fit_transform(y.values.reshape(-1,1))[:,0], axis=1)
    return stats_rescaled

def norm(X_i, X_m, W):
    """
    Compute norm between
    X_i and X_m as weighted
    by W.
    
    Parameters:
    -----------
    X_i : numpy.array
    1 x N (features)
    X_m : numpy.array
    K x N
    W : numpy.array
    1 x N
    """

    dX = X_m - X_i
    if W.ndim == 1:
        return (dX**2 * W).sum(1)
    else:
        return (dX.dot(W)*dX).sum(1)

def smallestm(d, m):
    """
    Find indices of smallest m numbers in 
    array d. Tied values included as well,
    returned index count may be greater than
    m. 
    
    Parameters:
    -----------
    d : numpy.array
    m : int
    
    Returns:
    --------
    smallest_idx : numpy.array
    """

    # Finds indices of the smallest m numbers in an array. Tied values are
    # included as well, so number of returned indices can be greater than m.

    # partition around (m+1)th order stat
    par_idx = pd.np.argpartition(d, m)

    if d[par_idx[:m]].max() < d[par_idx[m]]:  # m < (m+1)th
        return par_idx[:m]
    elif d[par_idx[m]] < d[par_idx[m+1:]].min():  # m+1 < (m+2)th
        return par_idx[:m+1]
    else:  # mth = (m+1)th = (m+2)th, so increment and recurse
        return smallestm(d, m+2)

def match(X_i, X_m, W, m):
    """
    Match the given unit X_i
    with up to m units from candidates
    X_m, while weighting the covariates
    with W.
    
    Parameters:
    -----------
    X_i : pandas.Series
    1 x N (N features)
    X_m : numpy.array
    K x N
    W : numpy.array
    1 x N
    m : int
    Minimum of matches returned.
    """
    d = norm(X_i, X_m, W)
    return smallestm(d, m)

def get_matches(control_words, treatment_words, X, matches=1):
    """
    Compute matches between treatment and control words
    based on matching covariates X. 
    
    Parameters:
    -----------
    control_words : [str]
    treatment_words : [str]
    X : pandas.DataFrame
    Covariate matrix. 
    matches : int
    Preferred number of minimum matches.
    """
    X_c = X.loc[control_words, :].values
    X_t = X.loc[treatment_words, :].values
    # compute weights according to inverse of covariate variance
    # ...because we want to upweight covariates with low variance
    # because they're more predictable
    W = 1 / X.var(axis=0).values
    matches_c = [match(X.loc[c].values, X_t, W, matches) for c in control_words]
    matches_t = [match(X.loc[t].values, X_c, W, matches) for t in treatment_words]
    return matches_c, matches_t

def get_logistic_decline_words():
    """
    Read logistic decline words and parameters
    from file.
    
    Returns:
    --------
    logistic_words : list
    logistic_params : pandas.DataFrame
    """
    logistic_file = '../../data/frequency/word_lists/2013_2016_logistic_growth_decline_words.csv'
    logistic_words = pd.read_csv(logistic_file, index_col=None).loc[:, 'word'].tolist()
    logistic_param_file = '../../data/frequency/2013_2016_tf_norm_logistic_params.tsv'
    logistic_params = pd.read_csv(logistic_param_file, sep='\t', index_col=0)
    return logistic_words, logistic_params

def get_piecewise_decline_words():
    """
    Read piecewise decline words and parameters
    from file.
    
    Returns:
    --------
    piecewise_words : list
    piecewise_params : pandas.DataFrame
    """
    piecewise_file = '../../data/frequency/word_lists/2013_2016_piecewise_growth_decline_words.csv'
    piecewise_decline = pd.read_csv(piecewise_file, index_col=None).loc[:, 'word'].tolist()
    piecewise_param_file = '../../data/frequency/2013_2016_tf_norm_log_2_piecewise_discrete.tsv'
    piecewise_params = pd.read_csv(piecewise_param_file, sep='\t', index_col=0)
    return piecewise_decline, piecewise_params

def get_growth_decline_words_and_params(data_dir='../../data/frequency/'):
    """
    Read all growth-decline words from file
    and also read the split point
    parameters.

    Parameters:
    ----------
    data_dir : str
    
    Returns:
    --------
    growth_decline_words : [str]
    split_points : pandas.Series
    """
    word_list_dir = os.path.join(data_dir, 'word_lists')
    growth_decline_piecewise = pd.read_csv(os.path.join(word_list_dir, '2013_2016_piecewise_growth_decline_words.csv'), index_col=None)['word'].tolist()
    growth_decline_logistic = pd.read_csv(os.path.join(word_list_dir, '2013_2016_logistic_growth_decline_words.csv'), index_col=None)['word'].tolist()
    growth_decline_logistic = list(set(growth_decline_logistic) - set(growth_decline_piecewise))
    growth_decline_words = growth_decline_piecewise + growth_decline_logistic
    piecewise_params = pd.read_csv(os.path.join(data_dir, '2013_2016_tf_norm_2_piecewise.tsv'), sep='\t', index_col=0).loc[growth_decline_piecewise, 'x0']
    logistic_params = pd.read_csv(os.path.join(data_dir, '2013_2016_tf_norm_logistic_params.tsv'), sep='\t', index_col=0).loc[growth_decline_logistic, 'loc']
    split_points = pd.concat([piecewise_params, logistic_params])
    N = 36
    MAX_SPLIT_POINT = N-2
    MIN_SPLIT_POINT = 1
    # dump the bad split points
    split_points = split_points[(split_points >= MIN_SPLIT_POINT) &
                                (split_points < MAX_SPLIT_POINT)]
    growth_decline_words = list(set(growth_decline_words) & set(split_points.index))
    return growth_decline_words, split_points

def get_growth_words(word_list_dir='../../data/frequency/word_lists'):
    """
    Read all growth words from file.

    Parameters:
    -----------
    word_list_dir : str
    
    Returns:
    --------
    growth_words : [str]
    """
    growth_word_file = os.path.join(word_list_dir, '2013_2016_growth_words_clean_final.csv')
    growth_words = pd.read_csv(growth_word_file, index_col=False).loc[:, 'word'].tolist()
    return growth_words

def get_success_fail_words(word_list_dir='../../data/frequency/word_lists/'):
    """
    Get list of success and fail words, and remove any intersect words.
    
    Parameters:
    -----------
    word_list_dir : str
    
    Returns:
    --------
    success_words : list
    fail_words : list
    split_points : pandas.Series
    """
    success_words = get_growth_words()
    fail_words, split_points = get_growth_decline_words_and_params()
    intersect_words = set(success_words) & set(fail_words)
    success_words = list(set(success_words) - intersect_words)
    fail_words = list(set(fail_words) - set(intersect_words))
    split_points = split_points.loc[fail_words]
    return success_words, fail_words, split_points

def get_success_words_final(word_list_dir='../../data/frequency/word_lists'):
    """
    Read all success words (final round of labelling) from file.
    
    Parameters:
    -----------
    word_list_dir : str
    
    Returns:
    --------
    success_words : list
    """
    success_word_file = os.path.join(word_list_dir, '2013_2016_success_words_final.tsv')
    success_words = pd.read_csv(success_word_file, sep='\t', index_col=0).index.tolist()
    return success_words

def get_fail_words_final(word_list_dir='../../data/frequency/word_lists'):
    """
    Read all fail words (final round of labelling) from file.
    
    Parameters:
    -----------
    word_list_dir : str
    
    Returns:
    --------
    fail_words : list
    """
    fail_word_file = os.path.join(word_list_dir, '2013_2016_fail_words_final.tsv')
    fail_words = pd.read_csv(fail_word_file, sep='\t', index_col=0).index.tolist()
    return fail_words

def get_all_covariates(data_dir='../../data/frequency/'):
    """
    Load covariates from file.
    
    Parameters:
    -----------
    data_dir : str
    
    Returns:
    --------
    covariates : [pandas.DataFrame]
    covariate_names : [str]
    """
    covariates = []
    tf_file_name = '2013_2016_tf_norm_log.tsv'
    C3_file_name = '2013_2016_3gram_residuals.tsv'
    DU_file_name = '2013_2016_user_diffusion.tsv'
    DS_file_name = '2013_2016_subreddit_diffusion.tsv'
    DT_file_name = '2013_2016_thread_diffusion.tsv'
    file_names = [tf_file_name, C3_file_name, DU_file_name, DS_file_name, DT_file_name]
    covariate_names = ['f', 'C3', 'DU', 'DS', 'DT']
    covariates = [pd.read_csv(os.path.join(data_dir, f), sep='\t', index_col=0) for f in file_names]
    return covariates, covariate_names