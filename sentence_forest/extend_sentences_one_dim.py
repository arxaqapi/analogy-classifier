import csv
import numpy as np

from embedd_sentences import embedd_row, glove_dict, EmbeddingError, load_vectors_fasttext

#an analogy a:b::c:d have 8 permutations (including the central permutation)
#central permutation is not valid for analogical sentences
VALID_PERMUTATION=4

def extendRow(a,b,c,d):
    row=[]
    row.extend(a)
    row.extend(b)
    row.extend(c)
    row.extend(d)
        
    return row  

# Only 4 permutations (remove central permutations
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
    row1=extendRow(a,b,c,d)    
    row2=extendRow(b,a,d,c)
    row3=extendRow(c,d,a,b)
    row4=extendRow(d,c,b,a)

    #Central permutations
    #row5=extendRow(a,c,b,d)
    #row6=extendRow(c,a,d,b)
    #row7=extendRow(b,d,a,c)
    #row8=extendRow(d,b,c,a)

    return row1, row2, row3, row4

    '''  
    return[ 
        [a, b, c, d],
        [a, c, b, d],
        [c, d, a, b],
        [c, a, d, b],
        [d, b, c, a],
        [d, c, b, a],
        [b, a, d, c],
        [b, d, a, c]
    ]
    '''    


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
        embedding_dict = glove_dict(embedding_size)
    elif word_embedding_used == 'fasttext':
        embedding_dict = load_vectors_fasttext()
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
                    embedding_dict=embedding_dict,
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
                y.extend([1] * VALID_PERMUTATION)
                # extend invalid
                bacd = bacd_invalid_extended(
                    embedded_row
                )
                cbad = cbad_invalid_extended(
                    embedded_row
                )
                X.extend(bacd)
                X.extend(cbad)
                y.extend([0] * VALID_PERMUTATION)
                y.extend([0] * VALID_PERMUTATION)
    print(f"[Log] - Skipped {skipped_quadruples} quadruples")
    return np.array(X), np.array(y)
