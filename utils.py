import csv
import numpy as np

def glove_dict(embedding_size):
    """Return the dictionnary containing each word vector

    Args:
        embedding_size (int): the size of the word vectors = 50, 100, 200 or 300

    Returns:
        dict: the dictionnar containing all word vectors of size embedding_size
    """
    embeddings_dict = {}
    with open("data/glove.6B/glove.6B." + str(embedding_size) + "d.txt", 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vec = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vec
    return embeddings_dict

def abcd_valid_extended(row, embedding_dict):
    a = embedding_dict[row[0].lower()]
    b = embedding_dict[row[1].lower()]
    c = embedding_dict[row[2].lower()]
    d = embedding_dict[row[3].lower()]
    return [
        np.stack([a, b, c, d]).T,
        np.stack([a, c, b, d]).T,
        np.stack([c, d, a, b]).T,
        np.stack([c, a, d, b]).T,
        np.stack([d, b, c, a]).T,
        np.stack([d, c, b, a]).T,
        np.stack([b, a, d, c]).T,
        np.stack([b, d, a, c]).T
    ]

def bacd_invalid_extended(row, embedding_dict):
    a = row[0]
    b = row[1]
    c = row[2]
    d = row[3]
    return abcd_valid_extended([b, a, c, d], embedding_dict)

def cbad_invalid_extended(row, embedding_dict):
    a = row[0]
    b = row[1]
    c = row[2]
    d = row[3]
    return abcd_valid_extended([c, b, a, d], embedding_dict)


def extendGoogleDataset(path, embedding_size=50):
    """
    - open the selected dataset (here the google dataset)
    - reads it, extends and embedd the data 
    - put the data in 2 variables, (X, y)
    path = path to the dataset
    """
    embeddings_dict = glove_dict(embedding_size)

    X = []
    y = []

    with open(path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        line_count = 0
        for row in csv_reader:
            if row[0] != ":":
                abcd = abcd_valid_extended(row, embeddings_dict)
                bacd = bacd_invalid_extended(row, embeddings_dict)
                cbad = cbad_invalid_extended(row, embeddings_dict)
                X.extend(abcd)
                X.extend(bacd)
                X.extend(cbad)
                y.extend([[1]] * 8)
                y.extend([[0]] * 8)
                y.extend([[0]] * 8)
            line_count += 1
            
    return (np.array(X), np.array(y))
