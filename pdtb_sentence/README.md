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


## Evaluating the network
First of all, we need to train our final model that can be used for inference.
```python
train_on_full_dataset(
    "semantic_sentence_database.csv",
    epochs=10,
    batch_size=128,
    embedding_size= 300 if WE_USED == 'fasttext' else EMBEDDING_SIZE,
    word_embedding_used=WE_USED,
    sentence_embedding_method=SE_USED,
    k=K
)
```
This function takes a dataset as input and the usual parameters (batch_size, embedding_size ...)
* The dataset is extended times 12 (4 positive and 8 negative examples) and embedded using the embedding technique detailed as the function parameter
* Once the dataset, denoted **X**, is ready, it is passed as an argument to the ```.fit(X)``` method
* The model is then saved for later use in inference

After the final model is fully trained, we can now start evaluating our model

We want to test our model on 2 types of sentence relations, *a:a:a:a* and *a:b::a:b*

* For this we generate an two evaluation dataset ```create_evaluation_dataset(n)``` of size ```n``` (here n = 20_000)
* Then we start the evaluation process ```load_and_predict(...)```:
  * The evaluation process takes the generated 2 datasets *aaaa* and *ababa*
  * Each dataset is embedded and reshaped to match the CNN input shape
  * We then infer the classes by running the dataset through the network and calculating the accuracy score

# References
* Almarwani, Nada, Hanan Aldarmaki, and Mona Diab. "Efficient Sentence Embedding using Discrete Cosine Transform."
* T. Mikolov, E. Grave, P. Bojanowski, C. Puhrsch, A. Joulin. "Advances in Pre-Training Distributed Word Representations"