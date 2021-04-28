
DIATOM - Disentangled Adversarial Neural Topic Model
===================

This is a repository for the NAACL 2021 paper:\
*A Disentangled Adversarial Neural Topic Model for Separating Opinions from Plots in User Reviews*


## Description ##
This repository provides:
- a PyTorch implementation of the **DIATOM** core architecture;
- an *extract* of the annotated sentences from the **MOBO** dataset.

The MOvie and BOok reviews dataset is a collection made up of movie and book reviews, paired with their related plots. The reviews come from different publicly available datasets: the Stanford's IMDB movie reviews [1], the GoodReads [2] and the Amazon reviews dataset [3].
With the help of 15 annotators, we further labeled more than 18,000 reviews' sentences (~6000 per corpus), marking the sentence polarity (*Positive*, *Negative*), or whether a sentence describes its corresponding movie/book *Plot*, or none of the above (*None*). 
In the `dataset` folder, we have shared an excerpt of the annotated sentences for each dataset.

Further details on the data annotation process and inter-annotator agreement are available in the paper.

[1]: [Learning word vectors for sentiment analysis](https://www.aclweb.org/anthology/P11-1015/), Maas et al., ACL11\
[2]: [Fine-grained spoiler detection from large-scale review corpora](https://www.aclweb.org/anthology/P19-1248/), Wan et al., ACL19\
[3]: [Image-based recommendations on styles and substitutes](https://dl.acm.org/doi/10.1145/2766462.2767755), McAuley et al., SIGIR15\
[4]: [MPST: A corpus of movie plot synopses with tags](https://www.aclweb.org/anthology/L18-1274/), Kar et al., LREC18


### Dataset Statistics ###
| Statistics  | IMDB | GoodReads | Amazon |
| ------------- | :---: | :---: | :---: |
| Number of Plots  | 1,131  | 150 | 100 |
| Number of Reviews | 25,836  | 83,852| 32,375|
| % Pos. reviews  | 0.46  | 0.33| 0.32|
| % Neg. reviews  | 0.54  | 0.50| 0.46 |
| % Neu. reviews  | 0.00  | 0.17 | 0.22 |
| Training set | 20,317  | 65,816| 25,883|
| Development set  | 2,965  | 9,007| 3,275|
| Test set  | 2,554  | 9,029| 3,217|
| Number of annotated sent. | 6,000  | 6,000 | 6,000|


## Requirements ##
- Python 3.x
- PyTorch >= 1.6.0
- [Gensim](https://radimrehurek.com/gensim/)
- [SentenceBERT](https://github.com/UKPLab/sentence-transformers)
- [Spacy](https://spacy.io/)
- tqdm


## Files ##
Current repository structure

./
- `diatom`: Core architecture of the DIATOM model
- `mobo_dataset`: Extract of the annotated sentences from the MOBO dataset

./diatom/
- `main_adv_vae.py`: Main file for training and test procedures
- `adversarial_vae_model.py`: DIATOM architecture and functions
- `vae_avitm_paper.py`: Basic VAE components adopted in DIATOM
- `sentiment_classifier.py`: Basic classifier used in the adversarial mechanism
- `topic_class.py`: Auxiliary topic class

./mobo_dataset/
- `Amazon_annotated_sentences_excerpt.json`
- `GoodReads_annotated_sentences_excerpt.json`
- `IMDB_annotated_sentences_excerpt.json`
