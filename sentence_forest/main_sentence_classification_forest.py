import sys
sys.path.append("../pdtb_sentence")
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
import numpy as np
from datetime import datetime
import os.path
import random
from extend_sentences_one_dim import extend_embedd_sentences
from utils import rnd, print_err
import gen_sentence_db
import pdtb_preprocess




K = 1
SEED = 7
FOLDS = 10


def train_forest(dataset, sentence_embedding_method, k, folds=FOLDS, embedding_size=50, n_estimators_=100, bootstrap_=True, max_depth_=None, max_features_='auto', min_samples_leaf_=1, min_samples_split_=2):
    if sentence_embedding_method == 'DCT':
        embedding_size *= k

    embedded_dataset, Y = dataset
    print_err("--- Training starts ---\n")
    random.seed()

    # KFold init
    kf = KFold(n_splits=folds, shuffle=True, random_state=5)
    print(f"[Log] - Shape of the dataset = {embedded_dataset.shape}")
    # Parameters

    fold = 1
    # Metrics
    confusion_matrices = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    acc_per_fold = []

    print_err("[Log] ---- START ----")
    for train_index, test_index in kf.split(embedded_dataset):

        X_train = embedded_dataset[train_index]
        y_train = Y[train_index]

        X_test = embedded_dataset[test_index]
        y_test = Y[test_index]

        # CREATE MODEL STRUCTURE
        forest = RandomForestClassifier(
            random_state=SEED,
            n_estimators=n_estimators_,
            bootstrap=bootstrap_,
            max_depth=max_depth_,
            max_features=max_features_,
            min_samples_leaf=min_samples_leaf_, min_samples_split=min_samples_split_,
            verbose=1
        )

        if (fold == 1):
            print(f"Forest model is {forest.get_params()}\n", flush=True)

        forest.fit(
            X_train,
            y_train
        )

        y_predicted = forest.predict(X_test)
        y_predicted = rnd(y_predicted)
        t_neg, f_pos, f_neg, t_pos = confusion_matrix(
            y_test, y_predicted
        ).ravel()
        confusion_matrices.append([t_neg, f_pos, f_neg, t_pos])
        acc_per_fold.append((t_neg+t_pos)/(t_neg+t_pos+f_pos+f_neg) * 100)

        print(f"Fold {fold}")
        print(
            f"t_neg = {t_neg} | f_pos = {f_pos} | f_neg = {f_neg} | t_pos = {t_pos}")
        print(f"Accuracy = {rnd(acc_per_fold[fold-1])}%", flush=True)
        report = classification_report(y_test, y_predicted)
        print(report)

        precision_scores.append(precision_score(y_test, y_predicted))
        recall_scores.append(recall_score(y_test, y_predicted))
        f1_scores.append(f1_score(y_test, y_predicted))
        fold += 1

    print(
        f"Average accuracy  : {rnd(np.mean(acc_per_fold))}% and standard deviation = {rnd(np.std(acc_per_fold))}")
    print(
        f"Average precision : {rnd(np.mean(precision_scores))} and standard deviation = {rnd(np.std(precision_scores))}")
    print(
        f"Average recall    : {rnd(np.mean(recall_scores))} and standard deviation = {rnd(np.std(recall_scores))}")
    print(
        f"Average f1        : {rnd(np.mean(f1_scores))} and standard deviation = {rnd(np.std(f1_scores))}")


##############################
# Program starts here
##############################
if len(sys.argv) < 5:
    print("Usage " + sys.argv[0] +
        " <dimension> <analogy_file> <embedding model>[fasttext|glove] <embedding medthod>[AVG|DCT] [k - integer] (for DCT)")
    exit(-1)


try:
    # setup the parameters required to build the NN Model
    embedding_size = int(sys.argv[1])

except:
    print("embedding size and epochs should be in numeric")

analogy_file = sys.argv[2]
word_embedding_used = sys.argv[3]
se_used = sys.argv[4]

if (se_used == 'DCT'):
    if len(sys.argv) == 6:
        try:
            # setup the parameters required to build the NN Model
            K = int(sys.argv[5])
        except:
            print_err("k should be in numeric")
    else:
        print_err("DCT is used, but k not specified. default k to 1")
        # K was set to 1 at the top of the program
if (word_embedding_used == 'fasttext'):

    print("Word dimension is fixed to 300")

print(
    f"[Log] - Parameters : folds = {FOLDS} | Word vector size = {embedding_size} | word_embedding_used ={word_embedding_used} | SE USED={se_used} | k={K}\n")

train_forest(
    dataset=extend_embedd_sentences(
        dataset_path=analogy_file,
        word_embedding_used=word_embedding_used,  # 'fasttext' or 'glove'
        sentence_embedding_method=se_used,  # AVG or DCT
        embedding_size=embedding_size,
        k=K
    ),
    sentence_embedding_method=se_used,
    k=K,
    folds=FOLDS,
    embedding_size=embedding_size
)
