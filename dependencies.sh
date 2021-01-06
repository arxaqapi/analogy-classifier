#!/bin/bash
wget http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip
wget http://download.tensorflow.org/data/questions-words.txt

wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip

mkdir -p data/glove.6B/
mkdir -p data/google/
mkdir -p data/fasttext/

mv glove.6B.zip data/glove.6B/
mv questions-words.txt data/google/
mv crawl-300d-2M.vec.zip data/fasttext/

unzip data/glove.6B/glove.6B.zip -d data/glove.6B/
unzip data/fasttext/crawl-300d-2M.vec.zip -d data/fasttext/

mkdir -p sentence/google_split/
mkdir -p pdtb_sentence/pdtb_split/