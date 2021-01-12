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


def extend_embedd_sentences(dataset_path, word_embedding_used, embedding_size, sentence_embedding_method, k):
    """This funtion should be called in main to start the business

    Args:
        path (string): path to the file containing the valid analogical sentences to extend

    Returns:
        tuple: X, y values containing the sentences and their corresponding y value (0 or 1)
    """
    print(f"[Log] - Extending and Embedding the {dataset_path}")
    X = []
    y = []
    
    if word_embedding_used == 'glove':
        embeddings_dict = glove_dict(embedding_size)
    elif word_embedding_used == 'fasttext':
        embeddings_dict = load_vectors_fasttext()
    else:
        raise ValueError("word_embedding_used should be 'glove' or 'fasttext' in extend_embedd_sentences()")
    
    skipped_quadruples = 0
    with open(dataset_path, 'r') as f:
        csv_file = csv.reader(f, delimiter='|')
        for row in csv_file:
            # Embedd a, b, c ,d
            try:
                embedded_row = embedd_row(
                    row,
                    word_embedding_used=word_embedding_used,
                    sentence_embedding_method=sentence_embedding_method,
                    embeddings_dict=embeddings_dict,
                    embedding_size=embedding_size,
                    k=k
                )
            except EmbeddingError as e:
                skipped_quadruples += 1
                # print(f"[Error] - {e}")
                pass
            else:
                # not executed if error
                abcd = abcd_valid_extended(
                    embedded_row
                )
                X.extend(abcd)
                y.extend([[1]] * 4) # 8 - 4
                # extend invalid
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
    print(f"[Log] - Skipped {skipped_quadruples} quadruples")
    return np.array(X), np.array(y)
