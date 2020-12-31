import io
import numpy as np


PREFIX = "fasttext/"

BIG_FT_DS = PREFIX + "crawl-300d-2M.vec"
SMALLER_FT_DS = PREFIX + "wiki-news-300d-1M.vec"


def load_vectors_mem(file_path):
    print(f"[Log] - Loading Embedding vectors ...")
    f = io.open(file_path, 'r', encoding='utf-8',
                newline='\n', errors='ignore')
    n, d = map(int, f.readline().split())
    print(f"N = {n} | d = {d}")
    embedding_dict = {}
    for line in f:
        tokens = line.rstrip().split(' ')
        embedding_dict[tokens[0]] = np.asarray(
            tokens[1:], float)  # map(float, tokens[1:])
    print(f"[Log] - Loading Embedding vectors finished")
    return embedding_dict


def load_vectors(file_path):
    print(f"[Log] - Loading Embedding vectors ...")
    embedding_dict = {}
    with open(file_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        n, d = map(int, f.readline().split())
        print(f"N = {n} | d = {d}")
        for line in f:
            values = line.rstrip().split(' ')
            embedding_dict[values[0]] = np.asarray(values[1:], "float32")
    print(f"[Log] - Loading Embedding vectors finished")
    return embedding_dict
