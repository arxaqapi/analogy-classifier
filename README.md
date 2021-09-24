# Analogical proportions between sentences: Theoretical aspects and preliminary experiments

This repository contains the code that supports the paper [Analogies Between Sentences: Theoretical Aspects - Preliminary Experiments](https://link.springer.com/chapter/10.1007/978-3-030-86772-0_1).

## Requirements
To get all the files and the correct folder structure, lauch the ```dependencies.sh``` script.

## Run the CNNs
#### Sentence classification task
* Go into the ```generated_sentences/``` directory
```$ cd generated_sentences/```
* Choose the embedding size at ```line 146``` in ```main_sentence_classification.py```
* Choose the number of ```epochs```, ```folds``` and ```batch_size``` in the train function at ```line 150```
* Start the main_sentence_classification.py script with the following command line:
```
$ python3 main_sentence_classification.py
```
#### Penn Discourse Treebank Sentence classification task
* Go into the ```pdtb_sentence/``` directory
```$ cd pdtb_sentence/```
* Please read and meet all the requirements layed out in the ```pdtb_sentence/README.md``` file
* Choose the embedding size at ```line 160``` in ```main_pdtb_sentence_classification.py```
* Choose the number of ```epochs```, ```folds``` and ```batch_size``` in the train function at ```line 164```
* Start the main_pdtb_sentence_classification.py script with the following command line:
```
$ python3 main_pdtb_sentence_classification.py
```

## Run the Random Forest classifier
* Check the ```sentence_forest/README_sentence_forest``` file.


## Cite us
```bibtex
@InProceedings{10.1007/978-3-030-86772-0_1,
    author="Afantenos, Stergos
        and Kunze, Tarek
        and Lim, Suryani
        and Prade, Henri
        and Richard, Gilles",
    editor="Vejnarov{\'a}, Ji{\v{r}}ina 
        and Wilson, Nic",
    title="Analogies Between Sentences: Theoretical Aspects - Preliminary Experiments",
    booktitle="Symbolic and Quantitative Approaches to Reasoning with Uncertainty",
    year="2021",
    publisher="Springer International Publishing",
    address="Cham",
    pages="3--18",
    abstract="Analogical proportions hold between 4 items a, b, c, d insofar as we can consider that ``a is to b as c is to d''. Such proportions are supposed to obey postulates, from which one can derive Boolean or numerical models that relate vector-based representations of items making a proportion. One basic postulate is the preservation of the proportion by permuting the central elements b and c. However this postulate becomes debatable in many cases when items are words or sentences. This paper proposes a weaker set of postulates based on internal reversal, from which new Boolean and numerical models are derived. The new system of postulates is used to extend a finite set of examples in a machine learning perspective. By embedding a whole sentence into a real-valued vector space, we tested the potential of these weaker postulates for classifying analogical sentences into valid and non-valid proportions. It is advocated that identifying analogical proportions between sentences may be of interest especially for checking discourse coherence, question-answering, argumentation and computational creativity. The proposed theoretical setting backed with promising preliminary experimental results also suggests the possibility of crossing a real-valued embedding with an ontology-based representation of words. This hybrid approach might provide some insights to automatically extract analogical proportions in natural language corpora.",
    isbn="978-3-030-86772-0"
}
```