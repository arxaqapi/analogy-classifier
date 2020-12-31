from load_vec import BIG_FT_DS, SMALLER_FT_DS, load_vectors

from scipy.fftpack import dct
import numpy as np
# test type 3 ??
sentvec = np.array([1, 2, 3, 5])
# Return the n == k coefficients
dct_k = dct(sentvec, type=2, n=15, norm='ortho', axis=-1)

# print(np.reshape(dct_k, (n * )))

var = [
    ['c0d1', 'c0d2', 'c0d3', 'c0d4', 'c0d5'],
    ['c1d1', 'c1d2', 'c1d3', 'c1d4', 'c1d5'],
    ['c2d1', 'c2d2', 'c2d3', 'c2d4', 'c2d5']
]

vec = np.array(var)
print(vec.shape)
new = np.reshape(vec, (vec.shape[0]*vec.shape[1]))
print(new)
print('-------------------------------')
k = 2
print(new[:vec.shape[1] * k])
print('-------------------------------')
print(np.ravel(var).shape)

# data = load_vectors(BIG_FT_DS)
# print(data['House'])
# print(data['House'].shape)

# def embedd_row -> apply sentence_dct to each sentence in row


def sentence_dct(sentence, k=2):
    # split into words
    for word in sentence:
        # embedd each word
        
        # for each dimension, dct on it
        # flatten k first DCT Terms
        pass
    # /!\ PADDING TO BEWARE OF
    pass
