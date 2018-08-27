# Word lists
This directory contains various attempts to define growth and decline words. 

We are interested in nonstandard words, which we define as words that would not plausibly be found in a newspaper.

We categorize the nonstandard words according to their process of generation, as follows:

- acronym (A)
- clipping (C)
- derivation (D)
- compound (K)
- onomatopoeia (O)
- respelling (R)
- slang (S)

THe word categories are provided in 2013_2016_word_categories.csv.

For growth words, we use the words defined in 2013_2016_growth_words_clean_final.csv, which were identified as having either a high Spearman's correlation coefficient or a high ratio of late to early frequency. All growth candidates are available in growth_word_candidates.csv.
For decline words, we use the words defined in 2013_2016_logistic_growth_decline_words.csv (fit to a logistic curve) and 2013_2016_piecewise_growth_decline_words.csv (fit to a piecewise linear trend for growth, decline). All decline candidates are available in decline_word_candidates.csv.

For validation, the two authors took a sample of 100 growth word candidates (based on high Spearman coefficient) and labelled them for nonstandard/standard in top_200_growth_interannotator_agreement.xlsx. We reached reasonable agreement and applied this ideal of nonstandard to the rest of the words.

Old versions of word lists can be found in old/