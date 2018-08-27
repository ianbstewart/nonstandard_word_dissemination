# How to do the things
I wrote a lot of scripts to process and analyze the data! Here's how to use some of those scripts.

## Install packages

We needed a few R packages and the R kernel for Jupyter notebooks. This requires sudo access.

        cd install
        bash install_R_notebooks.sh
        bash install_R_package.sh

## Data processing

### Download Reddit data
We got the Reddit data from the growing archive [here](https://files.pushshift.io/reddit/comments/). 

        cd data_processing
        bash download_reddit_comment_data.sh # downloads to /hg190/corpora/reddit_comment_data/
        
### Get metadata
We need this for counting users, threads, subreddits.

        cd data_processing
        bash extract_metadata.sh

### Get tf
Compute raw frequency for all words in raw corpus.
        
        cd frequency
        python get_tf.py
        bash combine_monthly_tf.sh
        python get_tf_norm.py
        
### Get vocab
Get top-100,000 vocab in corpus to normalize the long tail of uninteresting stuff.

        cd frequency
        bash get_top_vocab.sh

### Normalize data
- Get list of top subreddits, and get list of likely non-English subreddits.
- Filter non-English subreddits and spammers from corpus. (spammer/bot list downloaded [here](https://chenhaot.com/data/multi-community/README.txt) and stored at `data/metadata/bots.txt` and `data/metadata/spammers.txt`)
- Whitespace data, de-duplicate letters (more than 3 of the same consecutive character), replace user and sub names with dummy names.
- Get total tf, then compute top 100,000 words for vocabulary.
- Replace every non-vocabulary word with (1) NUM-UNK (if it contains only numbers) (2) CAP-UNK (if capitalized) or (3) CHAR-UNK (everything else). This step also produces the corresponding metadata file.

        cd social
        python reddit_community_comment_counts.py
        python get_top_k_communities.py
        python estimate_community_languages.py
        python get_non_english_communities.py
        
        cd data_processing
        bash filter_corpus.sh

        cd frequency
        bash get_monthly_tf.sh
        bash combine_monthly_tf.sh
        bash get_tf_norm.sh
        bash get_top_vocab.sh

        cd data_processing
        bash clean_corpus.sh

        bash normalize_corpus.sh

### Get social dissemination
For each word w, social dissemination is the unique number of users/subreddits/threads that used w normalized by the expected number of users/subreddits/threads.

For all scripts, need to manually set social var (user/subreddit/thread) inside the script.

Run social dissemination calculation in parallel over corpus chunks.

        cd social
        bash get_social_dissemination_from_raw_text.sh
        bash combine_social_dissemination_counts.sh
        bash get_log_stat.sh # convert to log to normalize distribution

(currently unused) This series of calculations computes the counts and then computes dissemination. It executes fine but the final social dissemination metrics were inaccurate, so we decided to do the whole process at once.

        cd social
        python get_social_word_counts.py
        python combine_social_word_counts.py
        python get_normalized_social_word_counts.py
        python get_social_var_comment_stats.py
        bash get_social_dissemination.sh
        bash combine_social_dissemination_counts.sh # and then combine

### Get ngram counts
To compute linguistic dissemination, you first need to get the full ngram frequencies, then for each word w count the number of ngrams containing w. We do this for each separate n-position:
for the string "that is cool af haha .", the ngrams containing "af" are different depending on the relative position.
(1-pos) "af haha ."
(2-pos) "cool af haha"
(3-pos) "is cool af"
So we have to count each case separately and then combine the counts. 

        bash get_ngram_counts.sh
        bash get_unique_ngrams_per_word.sh
        bash combine_unique_ngrams_per_word.sh
        bash combine_ngram_npos_counts.sh

To compute residuals (i.e. linguistic dissemination):

        bash get_ngram_residuals.sh

For all those scripts, you need to manually set the N values (we used N=2 and N=3).

### Get POS tags
We use the POS tagger hosted [here](https://github.com/brendano/ark-tweet-nlp/) to tag all words in the monthly corpora. Requires the tagger Github directory to be local at `scripts/ark-tweet-nlp-0.3.2/`.
This also takes a ton of disk space (4G per comment file) so be warned!!

        cd data_processing/
        bash get_pos_tags.sh
        
Then we have to count the tags and estimate the tag that applies to each word (based on highest relative frequency).

        cd frequency/
        bash get_tag_pcts.sh
        bash get_tag_estimates.sh
        bash combine_tag_pcts.sh

(currently unused) Count all POS ngram contexts for all words, based on the raw tagged files.

        cd frequency/
        bash get_pos_tag_ngram_context_counts.sh
        bash get_unique_tag_ngrams_per_word.sh
        bash get_pos_tag_ngram_residuals.sh

### Get growth and decline scores
To determine whether a word has significantly grown or declined over the period of interest, we need to measure their fit to a growth and decline trajectory. 

We measure "growth" with the Spearman correlation coefficient. 

        cd frequency/
        python get_growth_scores.py
        
We measure "decline" with (1) fit to a two-part piecewise linear function and (2) fit to a logistic function.

        cd frequency/
        python get_piecewise_fit_params_discrete.py
        python get_logistic_fit_params.py
        
We require growth candidates to have a growth score at or above the 85th percentile.

        cd frequency/
        python get_growth_candidates.py
        
We require logistic decline candidates to have a decline score (R2) at or above the 99th percentile and piecewise decline candidates to have a decline score (R2\*m1\*m2) at or above the 85th percentile. Logistic fits are more finicky than the piecewise fits.

        cd frequency/
        python get_decline_candidates.py

Next: manually annotate growth and decline candidates in `data/frequency/word_lists/` for standard/proper or not. The annotation might require extra contexts, which you can get from the normalized files.

        cd data_processing/
        bash get_sample_contexts_for_annotation_words.sh

Once you're done annotating, you should extract the growth and decline words into separate lists: 
- `data/frequency/word_lists/2013_2016_growth_words_clean_final.csv`
- `data/frequency/word_lists/2013_2016_logistic_growth_decline_words.csv`
- `data/frequency/word_lists/2013_2016_piecewise_growth_decline_words.csv`

### Descriptive stats
Before analysis, we generated some descriptive plots, including (1) example decline piecewise/logistic fits, (2) distribution of DL across POS groups, (3) survival curve, (4) best-fitting growth, logistic-decline and piecewise-decline words.

        cd frequency
        python plot_decline_examples.py 
        python plot_pos_DL_distribution.py
        python plot_split_point_distribution_survivors.py
        python plot_best_fitting_growth_decline_words.py
        
We also needed descriptive corpus statistics, like average comments per month.

        cd frequency
        python get_corpus_stats.py
        python get_meta_stats.py
        bash combine_corpus_stats.sh
        
As exploratory data, we needed examples of nonstandard words with high and low dissemination values.

        jupyter-notebook get_nonstandard_examples_high_low_covariates.ipynb

To demonstrate that DL is significantly different across POS groups, we need to run a one-way ANOVA on the word groups' mean DL values.

        cd frequency/
        bash anova_pos_DL_test.sh

## Analysis

Once you've manually annotated everything, it's time for analysis!

You can run all the tests at once as follows. It assumes that all outputs are written to `output/`.

        cd prediction/
        bash run_all_tests.sh
        
### Relative importance
First test is to compute relative importance of all covariates in explaining frequency change. 

        cd prediction/
        Rscript correlation_tests.R

### Causal inference

Second test is causal inference of each dissemination metric (treatment) on probability of growth versus decline (outcome). We compute the average dose response (continuous probability of growth/decline) for all treatment quantiles and plot the curves.

        cd prediction/
        Rscript get_average_dose_response_estimate_prob.R
        Rscript plot_average_dose_response_function.R 

### Binary classification
Third test is to predict whether a given word will grow or decline given dissemination metrics.

        cd prediction/
        python predict_growth_k_month_window.py
        python plot_success_k_month_window_lines.py

### POS tag robustness check
Subtask in third test: determine whether POS tags can explain difference in DL values between growth/decline words.

        cd prediction/
        python predict_growth_POS_tag.py --out_dir $OUT_DIR
        
        cd frequency/
        python plot_growth_vs_decline_pos_DL_distribution.py --out_dir $OUT_DIR

### Survival analysis
Last test is to build a Cox regression model to predict the time of "death" for all growth-decline words. This produces coefficients for all the covariates to compare their relative importance and compares the performance of separate regression models with different covariate sets. We do the easy stuff (coefficients, concordance) in Python and the hard stuff (deviance) in R.

        cd prediction/
        python survival_analysis_tests.py
        Rscript survival_analysis_tests.R
        python plot_concordance_scores.py
        
## Write-up

Last step is running the write-up code.

Oh, you thought that I had a script to generate the write-up automatically? Ha ha ha!

Most of the tables are manually interpreted from output written to file.

## Miscellaneous code

### Helper methods

I wrote a lot of helper methods to get stuff done!

        data_processing/data_handler.py
        prediction/prediction_helpers.py
        prediction/prediction_helpers.R
        prediction/hi_est_logit.R # Hirano-Imbens dose response function estimator with logistic regression
        
### Current unused code
        
Old plots:
    
        frequency/plot_frequency_vs_context_count_with_residuals.sh # plot frequency and context counts, then fit line between them
        frequency/plot_context_counts_with_examples.sh # plot example growth words on the frequency/context count plot
        frequency/visualize_context_diversity_with_examples.ipynb
        prediction/plot_prediction_accuracy.* # plot prediction accuracy from matched bootstrap prediction
        
Testing how stuff works:

        frequency/test_propensity_score_matching_R.ipynb # learning how to do propensity score matching
        prediction/causal_inference_continuous_treatment.ipynb # learning how to do continuous-treatment causal inference
        prediction/get_match_examples.* # get optimal match word pair examples
        social/visualize_dissemination_with_examples.ipynb # visualizing dissemination with growth word examples
        social/expected_user_diffusion.ipynb # testing expectation computing
        
Old prediction tasks:

        prediction/correlation_tests.py # correlating dissemination against frequency
        prediction/match_growth_growth_decline_words_k.R # match words on split point - k months
        prediction/bootstrap_matched_success_failure_prediction_non_differenced* # matching growth/decline words on frequency at split point, predicting growth versus decline
        prediction/get_average_dose_response_estimate.R # compute dose response estimate on frequency delta, rather than probability of growth
        
Wrangling data:

        frequency/zip_all_ngrams.sh
        data_processing/get_es_indices.sh # get active ElasticSearch indices for queries
        data_processing/elasticsearch_example.py # testing monthly aggregate counts
        data_processing/es_pool_example.py # example of pooled ES quewrwy
        data_processing/elasticsearch_date_range_scan.py # extracting comments that apply to date range