import csv
import io
import numpy as np
from scipy.fftpack import dct


class EmbeddingError(Exception):
    """Raised when the word cannot be embedded"""
    pass


def glove_dict(embedding_size):
    """Return the dictionnary containing each word vector

    Args:
        embedding_size (int): the size of the word vectors = 50, 100, 200 or 300

    Returns:
        dict: the dictionnar containing all word vectors of size embedding_size
    """
    print(f"[Log] - Loading Embedding vectors ...")
    embeddings_dict = {}
    with open("../data/glove.6B/glove.6B." + str(embedding_size) + "d.txt", 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vec = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vec
    print(f"[Log] - Loading Embedding vectors finished")
    return embeddings_dict


def load_vectors_fasttext():
    print(f"[Log] - Loading Embedding vectors ...")
    embedding_dict = {}
    with open("../data/fasttext/crawl-300d-2M.vec", 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        n, d = map(int, f.readline().split())
        print(f"N = {n} | d = {d}")
        for line in f:
            values = line.rstrip().split(' ')
            embedding_dict[values[0]] = np.asarray(values[1:], "float32")
    print(f"[Log] - Loading Embedding vectors finished")
    return embedding_dict


def pre_process_sentence(sentence, lower=False):
    safe_sentence = ""
    for e in sentence:
        if e in ["'", '"', ",", ";", ".", ':', '-', '/', "%", '$', '#', '&', '`', '(', ')', '{', '}', '!', '?', '<', '>']:
            safe_sentence += str(" " + e + " ")
        else:
            safe_sentence += e
    if lower:
        safe_sentence = safe_sentence.lower()
    safe_sentence = safe_sentence.strip().split(" ")
    safe_sentence = list(filter(None, safe_sentence))

    return safe_sentence


def avg_sent_to_vec(sentence, embeddings_dict, embedding_size, lower):
    safe_sentence = pre_process_sentence(sentence, lower=lower)

    vector = np.zeros(embedding_size)
    n_words = len(safe_sentence)
    for word in safe_sentence:
        try:
            vector += embeddings_dict[word]
        except KeyError:
            raise EmbeddingError(
                f"Could not embedd the word using 'AVG': {word}")
    return vector / n_words


def dct_sentence_to_vec(sentence, vector_dict, lower, k=2):
    safe_sentence = pre_process_sentence(sentence, lower=lower)

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


def embedd_row(row, word_embedding_used, sentence_embedding_method, embeddings_dict, embedding_size, k):
    if sentence_embedding_method == 'AVG' and word_embedding_used == 'glove':
        a = avg_sent_to_vec(row[0], embeddings_dict, embedding_size, lower=True)
        b = avg_sent_to_vec(row[1], embeddings_dict, embedding_size, lower=True)
        c = avg_sent_to_vec(row[2], embeddings_dict, embedding_size, lower=True)
        d = avg_sent_to_vec(row[3], embeddings_dict, embedding_size, lower=True)
    elif sentence_embedding_method == 'AVG' and word_embedding_used == 'fasttext':
        a = avg_sent_to_vec(row[0], embeddings_dict, embedding_size, lower=False)
        b = avg_sent_to_vec(row[1], embeddings_dict, embedding_size, lower=False)
        c = avg_sent_to_vec(row[2], embeddings_dict, embedding_size, lower=False)
        d = avg_sent_to_vec(row[3], embeddings_dict, embedding_size, lower=False)
    elif sentence_embedding_method == 'DCT' and word_embedding_used == 'glove':
        a = dct_sentence_to_vec(row[0], embeddings_dict, k=k, lower=True)
        b = dct_sentence_to_vec(row[1], embeddings_dict, k=k, lower=True)
        c = dct_sentence_to_vec(row[2], embeddings_dict, k=k, lower=True)
        d = dct_sentence_to_vec(row[3], embeddings_dict, k=k, lower=True)
    elif sentence_embedding_method == 'DCT' and word_embedding_used == 'fasttext':
        a = dct_sentence_to_vec(row[0], embeddings_dict, k=k, lower=False)
        b = dct_sentence_to_vec(row[1], embeddings_dict, k=k, lower=False)
        c = dct_sentence_to_vec(row[2], embeddings_dict, k=k, lower=False)
        d = dct_sentence_to_vec(row[3], embeddings_dict, k=k, lower=False)
    else:
        raise ValueError("sentence_embedding_method should be 'DCT' or 'AVG' in extend_embedd_sentences()\n and word_embedding_used should be 'glove' or 'fasttext' in extend_embedd_sentences()")
    assert a.shape == b.shape == c.shape == d.shape
    return [a, b, c, d]
