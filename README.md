# Nonstandard word dissemination
This repository accompanies the 2018 EMNLP paper concerning dissemination of nonstandard words online.

I. Stewart and J. Eisenstein. Making "fetch" happen: The influence of social and linguistic context on nonstandard word growth and decline. Proceedings of EMNLP 2018. [Paper here](https://arxiv.org/abs/1709.00345).

## Structure
- data/

Directory for all the term frequency, social and linguistic dissemination data to be computed.

- scripts/

Directory for all the data processing and analysis scripts.

- writing/

Directory for all the writing (NWAV, EMNLP, ML@GT).

## Before starting

We assume that you have already collected the raw Reddit comments data from [here](https://files.pushshift.io/reddit/comments/) for 2013-06 through 2016-05. 
These files are each about 5 GB zipped, so make sure your server/computer has enough space before downloading.

The scripts have the following dependencies:
- Python
    - scipy
    - numpy
    - pandas
    - matplotlib
    - sklearn
    - statsmodels
    - nltk
- R
    - relaimpo
    - causaldrf
