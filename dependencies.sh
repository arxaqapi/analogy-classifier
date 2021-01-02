#!/bin/bash
wget http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip
wget http://download.tensorflow.org/data/questions-words.txt

wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip
wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip

mkdir -p data/glove.6B/
mkdir -p data/google/
mkdir -p data/fasttext/

mv glove.6B.zip data/glove.6B/
mv questions-words.txt data/google/
mv wiki-news-300d-1M.vec.zip data/fasttext/
mv crawl-300d-2M.vec.zip data/fasttext/

unzip data/glove.6B/glove.6B.zip -d data/glove.6B/
unzip data/fasttext/wiki-news-300d-1M.vec.zip -d data/fasttext/
unzip data/fasttext/crawl-300d-2M.vec.zip -d data/fasttext/

mkdir -p sentence/temp/
mkdir -p pdtb_sentence/pdtb_split/