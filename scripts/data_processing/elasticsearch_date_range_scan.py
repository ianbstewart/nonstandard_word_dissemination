"""
Scan all comments from a provided week of the year.
"""

from datetime import datetime, timedelta
from elasticsearch import Elasticsearch,helpers
from collections import defaultdict
import codecs

EPOCH=datetime(1970,1,1)

def main():
    es = Elasticsearch(timeout=6000)
    print 'info ', es.info()
    index = 'reddit_comments-2015'
    start_date = datetime.strptime('2015 1 23', '%Y %w %W')
    end_date = start_date + timedelta(weeks=1)
    start_date_secs = (start_date - EPOCH).total_seconds()
    end_date_secs = (end_date - EPOCH).total_seconds()
    
    date_var = "created_utc"
    query = {
        "query": {
            "range" : {
                date_var : {
                    "gte" : start_date_secs,
                    "lt" :  end_date_secs,
                }
            }
        }
    }

    res = helpers.scan(es, index=index, query=query)
    ctr = 0
    for r in res:
        # print(r)
        ctr += 1
        if(ctr % 100000 == 0):
            print('got %d results'%(ctr))
    print('got %d results total'%(ctr))

if __name__ == '__main__':
    main()

