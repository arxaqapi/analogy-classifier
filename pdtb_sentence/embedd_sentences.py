import csv
import numpy as np

class EmbeddingError(Exception):
    """Raised when the word cannot be embedded"""
    pass


def glove_dict(embedding_size, path="data/glove.6B/"):
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


def sent_to_vec(sentence, embeddings_dict, embedding_size):
    safe_sentence = ""
    for e in sentence:
        # handle ’s
        if e in ["'", '"', ",", ";", ".", ':', '-', '/', "%", '$', '#', '&', '`', '(', ')', '{', '}', '!', '?', '<', '>']:
            safe_sentence += str(" " + e + " ")
        else:
            safe_sentence += e

    safe_sentence = safe_sentence.lower().strip().split(" ")
    safe_sentence = list(filter(None, safe_sentence))

    vector = np.zeros(embedding_size)  # size of the embedding
    n_words = len(safe_sentence)
    for word in safe_sentence:
        try:
            vector += embeddings_dict[word]
        except KeyError:
            raise EmbeddingError(f"Could not embedd the word {word}")
    return vector / n_words


def embedd_row(row, embeddings_dict, embedding_size):
    a = sent_to_vec(row[0], embeddings_dict, embedding_size)
    b = sent_to_vec(row[1], embeddings_dict, embedding_size)
    c = sent_to_vec(row[2], embeddings_dict, embedding_size)
    d = sent_to_vec(row[3], embeddings_dict, embedding_size)
    # Error should propagate
    return [a, b, c, d]
