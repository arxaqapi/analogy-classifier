#!/bin/bash
wget -P data/google/ http://download.tensorflow.org/data/questions-words.txt
wget -P data/glove.6B/ http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip
wget -P data/fasttext/ https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip

unzip data/glove.6B/glove.6B.zip -d data/glove.6B/
unzip data/fasttext/crawl-300d-2M.vec.zip -d data/fasttext/

mkdir -p generated_sentences/google_split/
mkdir -p generated_sentences/temp/
mkdir -p generated_sentences/reports/50/ generated_sentences/reports/100/ generated_sentences/reports/200/ generated_sentences/reports/300/

mkdir -p pdtb_sentence/temp/
mkdir -p pdtb_sentence/pdtb_split/
mkdir -p pdtb_sentence/pdtb_semantic_split/
mkdir -p pdtb_sentence/cnn_final_models/
mkdir -p pdtb_sentence/evaluation_files/
mkdir -p pdtb_sentence/reports/50/ pdtb_sentence/reports/100/ pdtb_sentence/reports/200/ pdtb_sentence/reports/300/
