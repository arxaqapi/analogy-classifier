import string
import re
from unicodedata import normalize
from collections import Counter
import csv
import os

#SOURCE:
#https://machinelearningmastery.com/prepare-french-english-dataset-machine-translation/
#European parliament translated sessions 2011
'''UTILITIES'''
# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, mode='rt', encoding='utf-8')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# split a loaded document into list of sentences
def to_sentences(doc):
	return doc.strip().split('\n')

# clean a list of lines
def clean_lines(lines):
	cleaned = list()
	# prepare regex for char filtering
	re_print = re.compile('[^%s]' % re.escape(string.printable))
	# prepare translation table for removing punctuation
	table = str.maketrans('', '', string.punctuation)
	for line in lines:
		# normalize unicode characters
		line = normalize('NFD', line).encode('ascii', 'ignore')
		line = line.decode('UTF-8')
		# tokenize on white space
		line = line.split()
		# convert to lower case
		line = [word.lower() for word in line]
		# remove punctuation from each token
		line = [word.translate(table) for word in line]
		# remove non-printable chars form each token
		line = [re_print.sub('', w) for w in line]
		# remove tokens with numbers in them
		line = [word for word in line if word.isalpha()]
		# store as string
		cleaned.append(' '.join(line))
	return cleaned

# create a frequency table for all words
def to_vocab(lines):
	vocab = Counter()
	for line in lines:
		tokens = line.split()
		vocab.update(tokens)
	return vocab

# remove all words with a frequency below a threshold
def trim_vocab(vocab, min_occurence):
	tokens = [k for k,c in vocab.items() if c >= min_occurence]
	return set(tokens)

# mark all OOV with "unk" for all lines
def update_dataset(lines, vocab):
	new_lines = list()
	for line in lines:
		new_tokens = list()
		for token in line.split():
			if token in vocab:
				new_tokens.append(token)
			else:
				new_tokens.append('unk')
		new_line = ' '.join(new_tokens)
		new_lines.append(new_line)
	return new_lines

def create_clean_reduced_list(filename): 
    """reduce the vocabulary to words with more than 5 occurrences"""
    doc = load_doc(filename)
    sentences = to_sentences(doc)
    sentences = clean_lines(sentences)
    # calculate vocabulary
    vocab = to_vocab(sentences)
    # reduce vocabulary - min num occurence = 5
    vocab = trim_vocab(vocab, 5)
    # mark out of vocabulary words
    sentences = update_dataset(sentences, vocab)
    # save updated dataset
    return sentences
            
def generate_analogy_from_pair_Fr_En(fr, en, number): 
	"""Create a single .csv of analogies between sentences Fr_En"""
	with open("analogy_Fr_En.csv", 'w') as f: 
		csv_file = csv.writer(f, delimiter='|')
		for i in range(number):
			row=(fr[i],en[i],fr[i+1],en[i+1])
			csv_file.writerow(row)
'''END UTILITIES'''
