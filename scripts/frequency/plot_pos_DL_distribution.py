"""
Generate boxplot of D_L values for each major POS group,
across the full vocabulary.
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import os

def main():
    parser = ArgumentParser()
    parser.add_argument('--data_dir', default='../../data')
    parser.add_argument('--out_dir', default='../../output')
    args = parser.parse_args()
    data_dir = args.data_dir
    out_dir = args.out_dir
    D_L = pd.read_csv(os.path.join(data_dir, 'frequency/2013_2016_3gram_residuals.tsv'), sep='\t', index_col=0)
    D_L_mean = D_L.mean(axis=1)
    tags = pd.read_csv(os.path.join(data_dir, 'frequency/2013_2016_tag_pcts.tsv'), sep='\t', index_col=0)
    tag_estimates = pd.DataFrame(tags.apply(lambda x: x.argmax(), axis=1), columns=['POS'])
    tag_meanings = pd.read_csv(os.path.join(data_dir, 'metadata/tag_meaning.tsv'), sep='\t', index_col=0)
    # make tag meanings printable
#     tag_meanings.loc[:, 'meaning'] = tag_meanings.loc[:, 'meaning'].apply(lambda x: x.replace('/', '\n').replace(' ', '\n'))
    tag_meanings.loc[:, 'meaning'] = tag_meanings.loc[:, 'meaning'].apply(lambda x: x.split('/')[0].replace(' ', '\n'))
    
    # restrict to top 10 most frequent tags
    # top_k = 10
    # top_tags = tag_estimates.loc[:, 'POS'].value_counts()[:top_k].index.tolist()
    # top_tags = ['N', '^', 'A', 'V', '!', 'R', 'D', 'E', '~']
#     top_tags = ['N', 'A', 'V', '!', 'R', '~']
    top_tags = ['!', 'A', 'G', 'N', 'R', 'V']
    top_k = len(top_tags)
    
    # collect data
    relevant_tags = tag_estimates[tag_estimates.loc[:, 'POS'].isin(top_tags)]
    box_data = []
    tag_list = []
    for t, group in relevant_tags.groupby('POS'):
        box_data_t = D_L_mean.loc[group.index.tolist()].dropna(inplace=False)
        box_data.append(box_data_t)
        tag_list.append(t)
    # sort in tag order
    tag_list, box_data = zip(*sorted(zip(tag_list, box_data), key=lambda x: x[0]))
    
    # PLOT PLOT PLOT
    xlabels = tag_meanings.loc[tag_list, 'meaning']
    X = pd.np.arange(top_k)
    xlabel = 'POS tag'
    ylabel = '$D^{L}$'
    med_col = 'k'
    box_col = '0.5' # gray
    label_size = 28
    xtick_size = 18
    ytick_size = 18
    tag_width = 1.4
    height = 3.5
    xtick_rotation = 20
    plt.figure(figsize=(tag_width*top_k, height))
    plt.boxplot(box_data, showfliers=True, patch_artist=True, 
                medianprops={'color' : med_col}, boxprops={'fc' : box_col}) # filling in boxes
    plt.xticks(X+1., xlabels, fontsize=xtick_size, rotation=xtick_rotation)
    plt.yticks(fontsize=ytick_size)
    plt.xlabel(xlabel, fontsize=label_size)
    plt.ylabel(ylabel, fontsize=label_size)
    # add horizontal line
    line_color = 'k'
    xlim = plt.xlim()
    plt.plot(xlim, [0,0], color=line_color)
    # make everything fit
    plt.tight_layout()
    out_file = os.path.join(out_dir, 'pos_DL_distribution.pdf')
    plt.savefig(out_file)
    
if __name__ == '__main__':
    main()
