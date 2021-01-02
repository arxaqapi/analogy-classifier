import csv
import io
import numpy as np
from scipy.fftpack import dct


class EmbeddingError(Exception):
    """Raised when the word cannot be embedded"""
    pass


def glove_dict(embedding_size, path):
    """Return the dictionnary containing each word vector

    Args:
        embedding_size (int): the size of the word vectors = 50, 100, 200 or 300

    Returns:
        dict: the dictionnar containing all word vectors of size embedding_size
    """
    embeddings_dict = {}
    # "data/glove.6B/glove.6B."
    with open(path + "glove.6B." + str(embedding_size) + "d.txt", 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vec = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vec
    return embeddings_dict


def load_vectors_fasttext(file_path):
    print(f"[Log] - Loading Embedding vectors ...")
    embedding_dict = {}
    with open(file_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        n, d = map(int, f.readline().split())
        print(f"N = {n} | d = {d}")
        for line in f:
            values = line.rstrip().split(' ')
            embedding_dict[values[0]] = np.asarray(values[1:], "float32")
    print(f"[Log] - Loading Embedding vectors finished")
    return embedding_dict


def load_vectors_fasttext_mem(file_path):
    print(f"[Log] - Loading Embedding vectors ...")
    f = io.open(file_path, 'r', encoding='utf-8',
                newline='\n', errors='ignore')
    n, d = map(int, f.readline().split())
    print(f"N = {n} | d = {d}")
    embedding_dict = {}
    for line in f:
        tokens = line.rstrip().split(' ')
        embedding_dict[tokens[0]] = np.asarray(
            tokens[1:], float)  # map(float, tokens[1:])
    print(f"[Log] - Loading Embedding vectors finished")
    return embedding_dict


def pre_process(sentence, lower=False):
    safe_sentence = ""
    for e in sentence:
        if e in ["'", '"', ",", ";", ".", ':', '-', '/', "%", '$', '#', '&', '`', '(', ')', '{', '}', '!', '?', '<', '>']:
            safe_sentence += str(" " + e + " ")
        else:
            safe_sentence += e

    safe_sentence = safe_sentence.strip().split(" ")
    safe_sentence = list(filter(None, safe_sentence))

    if lower:
        return safe_sentence.lower()
    else:
        return safe_sentence


def avg_sent_to_vec(sentence, embeddings_dict, embedding_size):
    safe_sentence = pre_process(sentence, lower=True)

    vector = np.zeros(embedding_size)
    n_words = len(safe_sentence)
    for word in safe_sentence:
        try:
            vector += embeddings_dict[word]
        except KeyError:
            raise EmbeddingError(
                f"Could not embedd the word using 'AVG': {word}")
    return vector / n_words


def dct_sentence(sentence, vector_dict, k=2):
    safe_sentence = pre_process(sentence, lower=False)

    embedded_word_matrix = []
    for word in safe_sentence:
        try:
            embedded_word_matrix.append(vector_dict[word])
        except KeyError:
            raise EmbeddingError(
                f"Could not embedd the word ising 'DCT': {word}")
    # DCT + first k DCT terms only
    dcted_vec_matrix = dct(
        embedded_word_matrix,
        type=2,
        n=k,
        norm='ortho',
        axis=0
    )[:k]
    # Flatten the array -> final sentence embedding
    return np.ravel(dcted_vec_matrix)


def embedd_row(row, embeddings_dict, embedding_size=100, k=2, type='AVG'):
    if type == 'AVG':
        a = avg_sent_to_vec(row[0], embeddings_dict, embedding_size)
        b = avg_sent_to_vec(row[1], embeddings_dict, embedding_size)
        c = avg_sent_to_vec(row[2], embeddings_dict, embedding_size)
        d = avg_sent_to_vec(row[3], embeddings_dict, embedding_size)
    elif type == 'DCT':
        a = dct_sentence(row[0], embeddings_dict, k)
        b = dct_sentence(row[1], embeddings_dict, k)
        c = dct_sentence(row[2], embeddings_dict, k)
        d = dct_sentence(row[3], embeddings_dict, k)
    # Error should propagate
    # print(f"Row check : {a.shape} == {b.shape} == {c.shape} == {d.shape}")
    assert a.shape == b.shape == c.shape == d.shape
    return [a, b, c, d]
