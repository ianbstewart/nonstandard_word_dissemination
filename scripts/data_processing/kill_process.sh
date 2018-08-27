# kill process by name
# PROCESS="python get_social_diffusion_parallel.py"
# PROCESS="python normalize_corpus.py"
# PROCESS="python get_ngram_counts.py"
# PROCESS="python get_monthly_frequency.py"
# PROCESS="python get_social_diffusion_from_text.py"
# PROCESS="python get_nearest_neighbors.py"
# PROCESS="bash get_pos_tags.sh"
# PROCESS="python get_ngram_counts.py"
# PROCESS="python get_unique_ngrams_per_word.py"
# PROCESS="ipykernel -f /nethome/istewart6/"
#PROCESS="python get_context_diversity_from_samples.py"
PROCESS="python get_piecewise_fit_params_discrete.py"
# PROCESS="python get_social_word_counts.py"
# PROCESS="bash get_word_sample.sh"
# PROCESS="wang2vec/word2vec -train"
# PROCESS=/usr/local/bin/python2.7
PROCESSES=$(ps aux | grep "$PROCESS" | awk '{print $2}')
echo $PROCESSES
kill $PROCESSES
