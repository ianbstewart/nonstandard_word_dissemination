"""
Example multithreaded ES querying 
with a shared pool of processes.
"""
from multiprocessing import Pool, Process
from elasticsearch import Elasticsearch
import argparse

def get_social_diffusion(word, social_var, 
#                          es, index, social_cutoff=0):
                         index, social_cutoff=0):
    """
    Compute social diffusion for a given word
    and social level, i.e. unique number of social
    vals per word.

    Parameters:
    -----------
    word : str
    social_var : str
    index : str
    social_cutoff : int
    Minimum number of word occurrences 
    for a social var to count.

    Returns:
    --------
    word_counts : pandas.DataFrame
    """
    # need a global ES instance because
    # it's not pickle-able, which we need
    # for multithreading
    print('bout to query ES for word %s and social var %s'%(word, social_var))
    es = GLOBAL_ES
    word_counts = defaultdict(list)
    unique_dates = set()
    query = { #aggregate query
        "query": {
        # "match_all":{}
          "match_phrase":{"body":word},
        },
        "aggs" : {
            "social_agg" : { 
                "date_histogram" : { 
                    "script":"(doc[\"created_utc\"].value) * 1000","interval" : "month"
                },
                "aggs": {
                    "social_counts" : {
                        "terms" : {
                            "field": social_var, "size" : 0,
                        }, 
                        "aggs" : {
                            "counting": { 
                                "sum" : {
                                    "script" : "_index['body.shingle']['%s'].tf()"%(word)
                                },
                            }
                        }
                    }
                }
            }
        }
    ,
     "size": 0
    }

     res = es.search(index=index, body=query)
    for bucket in res['aggregations']['social_agg']['buckets']:
        date = re.findall('201[0-9]-[0-9]{2}', bucket['key_as_string'])[0]
        unique_dates.add(date)
        counts = bucket['social_counts']['buckets']
        social_count = len([c for c in counts if 
                            c['counting']['value'] > social_cutoff])
        word_counts['social_diffusion'].append(social_count)
        word_counts['word'].append(word)
        word_counts['date'].append(date)
        word_counts['social_var'].append(social_var)
    word_counts = pd.DataFrame(word_counts)
    return word_counts

def query():

def main():
    parser = argparse.ArgumentParser()
    parser.add_arguments('--words', nargs='+', default=['af', 'doggo', 'adulting', 'ghosted'])
    parser.add_argument('--index', default='reddit_comments-2016')
    args = parser.parse_args()
    es = Elasticsearch()
    max_processes = 10
    pool = Pool(max_processes)

if __name__ == '__main__':
    main()
