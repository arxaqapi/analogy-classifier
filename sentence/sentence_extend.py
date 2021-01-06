import csv
import numpy as np

# from sentence_embedding import sent_to_vec


def abcd_valid_extended(row):
    """Takes a list of 4 strings as input,
        embedd each sentence then extends it to 8 valid analogies

    Args:
        row (list): a list of 4 strings to embedd
        embedding_dict (str, optional): the ebedding dictionnary used. Defaults to "../data/glove.6B/".
        embedding_size (int, optional): Size of the embedded word vector. Defaults to 100.

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
