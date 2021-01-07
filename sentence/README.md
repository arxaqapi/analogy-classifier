# Sentence classification task

## Usage
The following example generates the ```generated_sentences.csv``` file containing the sentence dataset based on the templates found in ```sentence_template.py```
```python
from google_sentence_gen import output_sentence_file
dataset = output_sentence()
```

### Sentence templates format
The sentences are in the form of a tuple containing one or two sentences (this number depends of the relationship category)

In each sentence, the word ```$``` will be replaced with a matching word contained in the google dataset and the quadruples are directly formed at the same time.

#### Special cases
There are three special cases handled by the sentence generator, the special cases are contained in brackets:
* ```[a,an]``` - In this case, we set this word to be ```an``` if the next word begins with a vowel sound, otherwise we set ```a``` as the word.
* ```[is,are]``` - In this case, we set the word to be ```is``` if the predecessor is singular, otherwise the word ```is``` will be set.
* For the opposite relationship category, the word in brackets will be detected and the sentence containing the antonym will be generated