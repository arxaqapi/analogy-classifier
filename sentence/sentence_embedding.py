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
    print("[Log] - Loading the embedding vectors...")
    embeddings_dict = {}
    # "data/glove.6B/glove.6B."
    with open(path + "glove.6B." + str(embedding_size) + "d.txt", 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vec = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vec
    print("[Log] - Loading finished")
    return embeddings_dict


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


def avg_sent_to_vec(sentence, embeddings_dict, embedding_size):
    safe_sentence = pre_process_sentence(sentence, lower=True)

    vector = np.zeros(embedding_size)
    n_words = len(safe_sentence)
    for word in safe_sentence:
        try:
            vector += embeddings_dict[word]
        except KeyError:
            raise EmbeddingError(
                f"Could not embedd the word using 'AVG': {word}")
    return vector / n_words


def embedd_row(row, embeddings_dict, embedding_size):
    a = avg_sent_to_vec(row[0], embeddings_dict, embedding_size)
    b = avg_sent_to_vec(row[1], embeddings_dict, embedding_size)
    c = avg_sent_to_vec(row[2], embeddings_dict, embedding_size)
    d = avg_sent_to_vec(row[3], embeddings_dict, embedding_size)
    assert a.shape == b.shape == c.shape == d.shape
    return [a, b, c, d]