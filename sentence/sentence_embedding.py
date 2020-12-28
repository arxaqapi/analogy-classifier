import numpy as np

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

def sent_to_vec(sentence, embedding_dict, embeding_size):
    sentence = sentence.strip().split(" ")
    vector = np.zeros(embeding_size) # size of the embedding
    n_words = len(sentence)
    for word in sentence:
        vector += embedding_dict[word]
    return vector / n_words