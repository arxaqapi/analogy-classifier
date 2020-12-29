# analogy-classifier

This repository contains a minified version of the word analogy classifier described in the following paper:

* Lim S., Prade H., Richard G. (2019) Solving Word Analogies: A Machine Learning Perspective. In: Kern-Isberner G., Ognjanović Z. (eds) Symbolic and Quantitative Approaches to Reasoning with Uncertainty. ECSQARU 2019. Lecture Notes in Computer Science, vol 11726. Springer, Cham. https://doi.org/10.1007/978-3-030-29765-7_20
* https://github.com/gillesirit/analogy

This version can be found in the ```word/``` directory

The sentence analogy classification task can be found in the ```sentence/```

## Details
The word embedding is done with GloVe:
* https://nlp.stanford.edu/projects/glove/

And the word dataset used is the Google's ```question-words.txt``` file
* http://download.tensorflow.org/data/questions-words.txt
## Requirements
To get all the files and the correct folder structure, lauch the ```dependencies.sh``` script.

## Run the CNN
#### Word classification task
* Go into the ```word/``` directory
```$ cd word/```
* Choose the embedding size at ```line 118``` in ```main_word_classification.py```
* Choose the number of ```epochs```, ```folds``` and ```batch_size``` in the train function at ```line 120```
* Start the main_word_classification.py script with the following command line:
```
$ python3 main_word_classification.py
```
#### Sentence classification task
* Go into the ```sentence/``` directory
```$ cd sentence/```
* Choose the embedding size at ```line 102``` in ```main_sentence_classification.py```
* Choose the number of ```epochs```, ```folds``` and ```batch_size``` in the train function at ```line 104```
* Start the main_sentence_classification.py script with the following command line:
```
$ python3 main_sentence_classification.py
```

## Example for the word classification task
* These are the final results obtained after running the CNN
```
Score per fold
------------------------------------------------------------------------
> Fold 1 - Loss: 0.012598558329045773 - Accuracy: 99.48833584785461%
------------------------------------------------------------------------
> Fold 2 - Loss: 0.013211878016591072 - Accuracy: 99.44356679916382%
------------------------------------------------------------------------
> Fold 3 - Loss: 0.014221073128283024 - Accuracy: 99.44570064544678%
------------------------------------------------------------------------
> Fold 4 - Loss: 0.010443086735904217 - Accuracy: 99.63757395744324%
------------------------------------------------------------------------
> Fold 5 - Loss: 0.01332341879606247 - Accuracy: 99.50112700462341%
------------------------------------------------------------------------
> Fold 6 - Loss: 0.009229166433215141 - Accuracy: 99.66741800308228%
------------------------------------------------------------------------
> Fold 7 - Loss: 0.00995730608701706 - Accuracy: 99.60558414459229%
------------------------------------------------------------------------
> Fold 8 - Loss: 0.011852254159748554 - Accuracy: 99.53949451446533%
------------------------------------------------------------------------
> Fold 9 - Loss: 0.016403404995799065 - Accuracy: 99.37320351600647%
------------------------------------------------------------------------
> Fold 10 - Loss: 0.008707469329237938 - Accuracy: 99.70365762710571%
------------------------------------------------------------------------
Average scores for all folds:
> Accuracy: 99.5405662059784 (+- 0.10345591582293846)
> Loss: 0.01199476160109043
------------------------------------------------------------------------
```
