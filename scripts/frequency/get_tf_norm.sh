# get normalized from raw frequency
RAW_TF=../../data/frequency/2013_2016_tf.tsv
VOCAB=ALL
(python get_tf_norm.py --raw_tf $RAW_TF --vocab $VOCAB)&