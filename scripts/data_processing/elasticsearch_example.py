'''
Elastic Search Query Example 2:
Monthly aggregation query
'''

from datetime import datetime
from elasticsearch import Elasticsearch,helpers
from collections import defaultdict
import codecs


es = Elasticsearch(timeout=6000)
print 'info ', es.info()

index = 'reddit_comments*2014*' #given
term = "actually" # given

query = { #agrregate query
  "query": {
    "match_all":{}   
  },
"aggs" : {"ovt" : { "date_histogram" : {"script":"(doc[\"created_utc\"].value) * 1000","interval" : "month"},
"aggs": { "subr":{  "terms": { "field": "subreddit"},

"aggs" : {"counting":{ "sum" :  {"script" : "_index['body.shingle2']['actually'].tf()"}
}}
}
}}}
,
 "size": 0
}


res = es.search(index=index, body = query)
print res

