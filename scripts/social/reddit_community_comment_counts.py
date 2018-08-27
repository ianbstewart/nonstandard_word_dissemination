"""
Get total number of comments per subreddit
over all time.
"""
import bz2
import json, os
from collections import defaultdict

if __name__ == '__main__':
    comment_years = ['2015', '2016']
    all_comment_files = []
    for y in comment_years:
        dir_y = '/mnt/new_hg190/corpora/reddit_comment_data/monthly_submission/%s/'%(y)
        comments = [os.path.join(dir_y, f) for f in os.listdir(dir_y)
                    if '.bz2' in f]
        all_comment_files += comments
    community_counts = defaultdict(int)
    ctr = 0
    cutoff = 1e10
    for comment_file_name in all_comment_files:
        print('processing comments from file %s'%(comment_file_name))
        comment_file = bz2.BZ2File(comment_file_name, 'r')
        for l in comment_file:
            comment = json.loads(l)
            community = comment['subreddit']
            community_counts[community] += 1
            ctr += 1
            if(ctr % 100000 == 0):
                print('processed %d comments'%(ctr))
            if(ctr >= cutoff):
                break
        if(ctr >= cutoff):
            break
    # write to file
    out_dir = '../data/'
    community_counts_sorted = sorted(community_counts.items(),
                                     key=lambda x: x[0])
    comment_dates = '2015_2016'
    with open(os.path.join(out_dir, '%s_community_counts.tsv'%(comment_dates)), 'w') as out_file:
        out_file.write('community\tcount\n')
        for community, count in community_counts_sorted:
            out_file.write('%s\t%d\n'%(community, count))
