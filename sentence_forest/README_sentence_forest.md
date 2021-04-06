Make sure you have :
1. run the dependencies as instructed for analogy sentence (dependencies.sh). 
2. run main_sentence_classification.py in the directory ../sentence to generate generated_sentences.csv.

The random forest sentence classification program requires 4 parameters:
parameter 1: name of the file which contain the data (analogy sentences)
parameter 2: the vector size of the word embedding model (50, 100, 200 or 300) for Glove. This parameter is ignored when using fasttext word embedding
parameter 3: the model of embedding (glove or fasttext). The vector size is 300 and cannot be changed.
parameter 4: sentence embedding model. AVG for average or DCT for discrete cosine transform.
parameter 5: the value of k (integer), default to 1. Only useful for DCT. Note that the sentence vector size is the size of word embedding * k; for example, if the vector size used is 50, and k is 2, then the final vector size is 100.

Use output redirection to store the results in a file. 

Example use
The following example uses vector size 50, generated_sentences.csv analogy file, glove word embedding, average sentence embedding and stores the output to result_forest_google_glove_50_avg.txt

   python3.6 main_sentence_classification_forest.py 50 ../generated_sentences/generated_sentences.csv glove AVG > result_forest_google_glove_50_avg.txt
 
 Penn Database requires a licence, so it is unavailable for public access.

