# pdtb-sentence

## Prerequisites
* The PDTB versin used here is the 2.0 PDTB corpus
* Be sure to execute the ```../dependencies.sh``` script before starting anything.
* To be able to work, you have to copy all the 24 directories containing the pre-processed data of the PDTB database into the ```pdtb/``` folder (there should be a perl script pre-processing the raw data into a bunch of folders named from ```00/``` to ```24/```, the files inside these folders should have the ```.pipe``` extension)

```
.
├── pdtb                # PDTB folder structure
│   ├── 00
│   ├── 01
│   ├── ...
│   ├── 23
│   └── 24
```
## Files
* The dataset of quadruples is generated dynamically via pdtb_preprocess.py leading to a csv file explicit_sentence_database.csv.
* main_pdtb_sentence_classification.py uses explicit_sentence_database.csv as training set.

## Usage
* Choose the embedding size, the k value if Discrete Cosine Transform is used and the sentence embedding method used
```python
EMBEDDING_SIZE = 300
K = 6
SE_USED = 'AVG' # 'AVG' or 'DCT'
```
* Choose the number of ```epochs```, ```folds``` and ```batch_size``` in the train function at ```line 164```
* Start the main_pdtb_sentence_classification.py script with the following command line:
```
$ python3 main_pdtb_sentence_classification.py
```

## Details
If a specific word w belonging to a sentence s cannot be embedded using the GloVe database (i.e. the word w does not have a precomputed vector), we throw the whole sentence away and we do not use sentence s anymore to build quadruples of analogical sentences.


# References
* Almarwani, Nada, Hanan Aldarmaki, and Mona Diab. "Efficient Sentence Embedding using Discrete Cosine Transform."
* T. Mikolov, E. Grave, P. Bojanowski, C. Puhrsch, A. Joulin. "Advances in Pre-Training Distributed Word Representations"