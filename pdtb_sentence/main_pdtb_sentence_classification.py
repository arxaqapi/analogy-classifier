import pdtb_preprocess
import gen_sentence_db
from extend_sentences import extend_embedd_sentences

import random
import os.path
from datetime import datetime

import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, BatchNormalization, Flatten, Dense, Activation
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score

import tensorflow as tf


def cnn_model(shape=(50, 4, 1)):
    model = Sequential([
        Conv2D(
            filters=128,
            kernel_size=(1, 2),
            strides=(1, 2),
            input_shape=shape
        ),
        BatchNormalization(axis=-1),
        Activation('relu'),
        Conv2D(
            filters=64,
            kernel_size=(2, 2),
            strides=(2, 2)
        ),
        BatchNormalization(axis=-1),
        Activation('relu'),
        Flatten(),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        loss='binary_crossentropy',
        metrics=['accuracy'],
        optimizer='adam'
    )
    return model


def save(model, name):
    today = datetime.now()
    model.save(
        "cnn_model/" +
        str(today.strftime("%d_%m_%Y__%H_%M_%S_")) +
        name +
        '.keras')


def train(dataset, sentence_embedding_method, k, epochs=10, batch_size=32, folds=10, embedding_size=50):
    if sentence_embedding_method == 'DCT':
        embedding_size *= k

    embedded_dataset, Y = dataset

    print(
        f"[Log] - Parameters : epochs = {epochs} | batch_size = {batch_size} |folds = {folds} | Word vector size = {embedding_size}")
    random.seed()

    # KFold init
    kf = KFold(n_splits=folds, shuffle=True, random_state=5)

    print(f"[Log] - Pre-Shape of the dataset = {embedded_dataset.shape}")
    # Prepare data for convolutional layer
    embedded_dataset = np.reshape(
        embedded_dataset,
        (embedded_dataset.shape[0], embedding_size, 4, 1)
    )
    print(f"[Log] - Shape of the dataset = {embedded_dataset.shape}")
    # Parameters
    input_shape = embedded_dataset[0].shape
    fold = 1
    verbosity = 1
    # Metrics
    confusion_matrices = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    acc_per_fold = []

    print("[Log] ---- START ----")
    for train_index, test_index in kf.split(embedded_dataset):

        X_train = embedded_dataset[train_index]
        y_train = Y[train_index]

        X_test = embedded_dataset[test_index]
        y_test = Y[test_index]

        cnn = cnn_model(input_shape)

        # history =
        cnn.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbosity
            # callbacks=callbacks
        )
        # Metrics
        scores = cnn.evaluate(
            X_test,
            y_test,
            verbose=verbosity
        )
        y_predicted = cnn.predict(X_test)
        y_predicted = np.around(np.array(y_predicted), 0)
        t_neg, f_pos, f_neg, t_pos = confusion_matrix(
            y_test, y_predicted
        ).ravel()
        confusion_matrices.append([t_neg, f_pos, f_neg, t_pos])
        print(
            f"t_neg = {t_neg} | f_pos = {f_pos} | f_neg = {f_neg} | t_pos = {t_pos}")

        report = classification_report(y_test, y_predicted)
        print(report)

        precision_scores.append(precision_score(y_test, y_predicted))
        recall_scores.append(recall_score(y_test, y_predicted))
        f1_scores.append(f1_score(y_test, y_predicted))
        acc_per_fold.append(scores[1] * 100)

        save(cnn, f"fold_{fold}")
        fold += 1
    print(
        f"Cross folds average accuracy for {epochs} epochs : {np.mean(acc_per_fold)} and standart deviation = {np.std(acc_per_fold)}")
    print(
        f"Average precision : {np.mean(precision_scores)} and standart deviation = {np.std(precision_scores)}")
    print(
        f"Average recall : {np.mean(recall_scores)} and standart deviation = {np.std(recall_scores)}")
    print(
        f"Average f1 : {np.mean(f1_scores)} and standart deviation = {np.std(f1_scores)}")


PATH_TO_CSV = "pdtb/pdtb_sentences.csv"

if not os.path.isfile("explicit_sentence_database.csv"):
    # .pipe -> single csv
    pdtb_preprocess.create_single_csv_from_pdtb(PATH_TO_CSV)
    # split into sept csv's
    data_dict = pdtb_preprocess.split_single_csv_into_relation_type_files(
        PATH_TO_CSV)
    # generate explicit sentence database
    gen_sentence_db.randomly_generate_n_sentence_quadruples(
        data_dict,
        20000
    )
    print("[Log] - Initialization finished")

EMBEDDING_SIZE = 300
K = 6
SE_USED = 'AVG' # 'AVG' or 'DCT'

train(
    dataset = extend_embedd_sentences(
        dataset_path="explicit_sentence_database.csv",
        word_embedding_used='fasttext',  # 'fasttext' or 'glove'
        sentence_embedding_method=SE_USED,
        embedding_size=EMBEDDING_SIZE,
        k=K
    ),
    epochs=10,
    batch_size=16,
    folds=10,
    embedding_size=EMBEDDING_SIZE,
    sentence_embedding_method=SE_USED,
    k=K
)
