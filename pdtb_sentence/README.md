# pdtb-sentence

## Prerequisites
* The PDTB versin used here is the 2.0 PDTB corpus
* Be sure to execute the ```../dependencies.sh``` script before starting anything
* To be able to work, you have to copy all the 24 directories containing the pre-processed data of the PDTB database into the ```pdtb/``` folder (there should be a perl script pre-processing the raw data into a bunch of folders named from ```00/``` to ```24/```, the files inside these folders should have the ```.pipe``` extension)

    .
    ├── pdtb                    # PDTB folder structure
    │   ├── 00
    │   ├── 01
    │   ├── ...
    │   ├── 23
    │   └── 24

## Usage
* Choose the embedding size and the path where the dataset used in the cnn will be stored
```
EMBEDDING_SIZE = 50
PATH_TO_CSV = "pdtb/pdtb_sentences.csv"
```
* Choose the number of ```epochs```, ```folds``` and ```batch_size``` in the train function at ```line 118```
* Start the main_pdtb_sentence_classification.py script with the following command line:
```
$ python3 main_pdtb_sentence_classification.py
```

## Details
If a specific word cannot be embedded using the GloVe database (i.e. the word does not have a precomputed vector), We throw that example away, so we get rid of the whole line containing the quadruples of sentences

### Relation types (R)
* Explicit - Explicit Relations
* Implicit - Implicit Relations
* AltLex - Alternate Lexicalizations
* EntRel - Entity Relations
* NoRel - No Relations
