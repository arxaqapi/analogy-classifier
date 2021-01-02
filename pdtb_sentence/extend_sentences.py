import csv
import numpy as np

from embedd_sentences import embedd_row, glove_dict, EmbeddingError, load_vectors_fasttext


def abcd_valid_extended(row):
    """Takes a list of 4 strings as input,
        Extends each row to 8 valid analogies

    Args:
        row (list): a list of 4 strings to embedd

    Returns:
        list: a list containing the 8 valid analogies sentence lists
    """
    a = row[0]
    b = row[1]
    c = row[2]
    d = row[3]
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


def bacd_invalid_extended(row):
    a = row[0]
    b = row[1]
    c = row[2]
    d = row[3]
    return abcd_valid_extended([b, a, c, d])


def cbad_invalid_extended(row):
    a = row[0]
    b = row[1]
    c = row[2]
    d = row[3]
    return abcd_valid_extended([c, b, a, d])


def extend_embedd_sentences(path, embedding_size, word_vectors_path, k, type='AVG'):
    """This funtion should be called in main to start the business

    Args:
        path (string): path to the file containing the valid analogical sentences to extend

    Returns:
        tuple: X, y values containing the sentences and their corresponding y value (0 or 1)
    """
    print(f"[Log] - Extending and Embedding the {path}")
    X = []
    y = []
    if type == 'AVG':
        embeddings_dict = glove_dict(embedding_size, word_vectors_path)#"../data/glove.6B/")
    elif type == 'DCT':
        embeddings_dict = load_vectors_fasttext(word_vectors_path)
    else:
        raise ValueError("Type should be 'AVG' or 'DCT' in extend_embedd_sentences()")
    # if it works, do not write to file
    with open(path, 'r') as f:
        csv_file = csv.reader(f, delimiter='|')
        for row in csv_file:
            # Embedd a, b, c ,d
            try:
                # ZEAZEAE
                if type == 'AVG':
                    embedded_row = embedd_row(row, embeddings_dict, embedding_size=embedding_size, type='AVG')
                elif type == 'DCT':
                    embedded_row = embedd_row(row, embeddings_dict, k=k, type='DCT')
            except EmbeddingError as e:
                # print(f"[Error] - {e}")
                pass
            else:
                # not executed if error
                abcd = abcd_valid_extended(
                    embedded_row
                )
                X.extend(abcd)
                y.extend([[1]] * 8)
                # extend invalid
                bacd = bacd_invalid_extended(
                    embedded_row
                )
                cbad = cbad_invalid_extended(
                    embedded_row
                )
                X.extend(bacd)
                X.extend(cbad)
                y.extend([[0]] * 8)
                y.extend([[0]] * 8)
    return np.array(X), np.array(y)
