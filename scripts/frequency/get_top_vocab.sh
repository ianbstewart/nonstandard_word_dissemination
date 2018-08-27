DATA_DIR=../../data/frequency
TF=$DATA_DIR/2013_2016_tf_norm.tsv
K=100000
python get_top_vocab.py --tf $TF --top_k $K