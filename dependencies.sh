#!/bin/bash
wget http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip
wget http://download.tensorflow.org/data/questions-words.txt

mkdir -p data/glove.6B/
mkdir -p data/google/

mv glove.6B.zip data/glove.6B/
mv questions-words.txt data/google/

unzip data/glove.6B/glove.6B.zip -d data/glove.6B/