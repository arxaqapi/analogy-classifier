import csv
import numpy as np

from sentence_embedding import embedd_row, EmbeddingError


def abcd_valid_extended(row):
    """Takes a list of 4 strings as input,
        extends it to 8 or 4 valid analogies

    Args:
        row (list): a list of 4 strings

    Returns:
        list: a list containing the 8 or 4 valid analogies sentence lists
    """
    a = row[0]
    b = row[1]
    c = row[2]
    d = row[3]
    return [
        np.stack([a, b, c, d]).T, #
        np.stack([b, a, d, c]).T, #
        np.stack([c, d, a, b]).T, #
        np.stack([d, c, b, a]).T, #
        # np.stack([a, c, b, d]).T,
        # np.stack([d, b, c, a]).T,
        # np.stack([c, a, d, b]).T,
        # np.stack([b, d, a, c]).T
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


def extend_embedd_subset(dataset, embedding_dict, embedding_size, sentence_embedding_method, word_embedding_used, k):
    X = []
    y = []
    skipped_quadruples = 0
    with open(dataset, 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f, delimiter='|')
        for row in csv_reader:
            try:
                embedded_row = embedd_row(
                    row,
                    embedding_dict=embedding_dict,
                    embedding_size=embedding_size,
                    sentence_embedding_method=sentence_embedding_method,
                    word_embedding_used=word_embedding_used,
                    k=k
                )
            except EmbeddingError:
                skipped_quadruples += 1
            else:
                # not executed if error
                abcd = abcd_valid_extended(
                    embedded_row
                )
                X.extend(abcd)
                y.extend([[1]] * 4) # 8 - 4
                bacd = bacd_invalid_extended(
                    embedded_row
                )
                cbad = cbad_invalid_extended(
                    embedded_row
                )
                X.extend(bacd)
                X.extend(cbad)
                y.extend([[0]] * 4) # 8 - 4
                y.extend([[0]] * 4) # 8 - 4
    X = np.array(X)
    X = np.reshape(X, (X.shape[0], embedding_size, 4, 1))
    return X, np.array(y)