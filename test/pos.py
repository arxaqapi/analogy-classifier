from __future__ import absolute_import, division, unicode_literals
from scipy.fftpack import dct
import senteval

import sys
import io
import numpy as np
import logging


# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data'
# PATH_TO_VEC = 'glove/glove.840B.300d.txt'
PATH_TO_VEC = 'fasttext/crawl-300d-2M.vec'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)


# Create dictionary

def create_dictionary(sentences, threshold=0):
    words = {}
    for s in sentences:
        for word in s:
            words[word] = words.get(word, 0) + 1

    if threshold > 0:
        newwords = {}
        for word in words:
            if words[word] >= threshold:
                newwords[word] = words[word]
        words = newwords
    words['<s>'] = 1e9 + 4
    words['</s>'] = 1e9 + 3
    words['<p>'] = 1e9 + 2

    sorted_words = sorted(words.items(), key=lambda x: -x[1])  # inverse sort
    id2word = []
    word2id = {}
    for i, (w, _) in enumerate(sorted_words):
        id2word.append(w)
        word2id[w] = i

    return id2word, word2id

# Get word vectors from vocabulary (glove, word2vec, fasttext ..)


def get_wordvec(path_to_vec, word2id):
    word_vec = {}

    with io.open(path_to_vec, 'r', encoding='utf-8') as f:
        # if word2vec or fasttext file : skip first line "next(f)"
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word2id:
                word_vec[word] = np.fromstring(vec, sep=' ')

    logging.info('Found {0} words with word vectors, out of \
        {1} words'.format(len(word_vec), len(word2id)))
    return word_vec


# SentEval prepare and batcher
def prepare(params, samples):
    _, params.word2id = create_dictionary(samples)
    params.word_vec = get_wordvec(PATH_TO_VEC, params.word2id)
    params.wvec_dim = 300
    return


def batcher(params, batch):
    batch = [sent if sent != [] else ['.'] for sent in batch]
    embeddings = []

    for sent in batch:
        sentvec = []
        for word in sent:
            if word in params.word_vec:
                sentvec.append(params.word_vec[word])
        if not sentvec:
            vec = np.zeros(params.wvec_dim)
            sentvec.append(vec)

        if (sentvec.shape[0] < int(sys.argv[1]):
            sentvec=np.reshape(
                dct(
                    sentvec,
                    n=int(sys.argv[1]),
                    norm='ortho',
                    axis=0
                )[:int(sys.argv[1]), :],
                (int(sys.argv[1])*sentvec.shape[1],))
        else:
            sentvec=np.reshape(
                dct(
                    sentvec,
                    norm='ortho',
                    axis=0)[:int(sys.argv[1]), :],
                    (int(sys.argv[1])*sentvec.shape[1],)
                    )

        embeddings.append(sentvec)

    embeddings=np.vstack(embeddings)
    return embeddings


# Set params for SentEval
# params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
# params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
#                                 'tenacity': 3, 'epoch_size': 2}
params_senteval_Dep_NP={'task_path': PATH_TO_DATA,
    'usepytorch': True, 'kfold': 10}
params_senteval_Dep_NP['classifier']={
    'nhid': 0,
    'optim': 'adam',
    'batch_size': 64,
    'tenacity': 5,
    'epoch_size': 4
}
# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    se=senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks=['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
    'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
    'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
    'Length', 'WordContent', 'Depth', 'TopConstituents',
    'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
    'OddManOut', 'CoordinationInversion'
    ]
    results=se.eval(transfer_tasks)
    print(results)
