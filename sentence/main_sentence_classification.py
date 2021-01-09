from google_sentence_gen import extend_embedd_subset, output_sentence_file
from sentence_embedding import glove_dict

import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Flatten, Dense, Activation
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score

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


def rnd(n):
    return np.around(n, 3)


def train(dataset_path, epochs=10, batch_size=32, folds=10, embedding_size=50):
    report_name = f"report_e{epochs}_f{folds}_wv{embedding_size}.txt"
    with open(report_name, 'w') as f:
        f.write("--- Training starts ---\n")
        f.write(f"- Parameters : epochs = {epochs} | batch_size = {batch_size} |folds = {folds} | Word vector size = {embedding_size}\n")
    # KFold init
    random.seed()
    kf = KFold(n_splits=folds, shuffle=True, random_state=5)

    dataset = pd.read_csv(dataset_path, header=None, delimiter='|').to_numpy()
    input_shape = (embedding_size, 4, 1)
    fold = 1
    verbosity = 1
    embedding_dict = glove_dict(embedding_size, "../data/glove.6B/")
    # TODO
    # pre-process file / array
    confusion_matrices = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    acc_per_fold = []
    target_names = ['invalid analogy', 'valid analogy']
    print("[Log] ---- START ----")
    for train_index, test_index in kf.split(dataset):

        train_file = "temp/train_" + str(fold) + ".csv"
        pd.DataFrame(dataset[train_index]).to_csv(train_file, sep='|', index=False, header=None)
        X_train, y_train = extend_embedd_subset(train_file, embedding_dict, embedding_size)

        test_file = "temp/test_" + str(fold) + ".csv"
        pd.DataFrame(dataset[test_index]).to_csv(test_file, sep='|', index=False, header=None)
        X_test, y_test = extend_embedd_subset(test_file, embedding_dict, embedding_size)

        cnn = cnn_model(input_shape)

        cnn.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbosity
        )

        scores = cnn.evaluate(
            X_test,
            y_test,
            verbose=verbosity
        )
        # Metrics
        y_predicted = np.around(cnn.predict(X_test), 0)
        t_neg, f_pos, f_neg, t_pos = confusion_matrix(
            y_test, y_predicted
        ).ravel()
        confusion_matrices.append([t_neg, f_pos, f_neg, t_pos])
        report = classification_report(y_test, y_predicted, target_names=target_names)

        precision_scores.append(precision_score(y_test, y_predicted))
        recall_scores.append(recall_score(y_test, y_predicted))
        f1_scores.append(f1_score(y_test, y_predicted))
        acc_per_fold.append(scores[1] * 100)
        with open(report_name, 'a') as f:
            f.write("-"*75)
            f.write(f"\n  - Fold n°{fold}:\n")
            f.write(f"t_neg = {t_neg} | f_pos = {f_pos} | f_neg = {f_neg} | t_pos = {t_pos}\n")
            f.writelines(report)
        save(cnn, f"fold_{fold}\n")
        fold += 1
    with open(report_name, 'a', encoding='utf-8') as f:
        f.write(f"\n--- Total results after {folds} folds for vector size = {embedding_size} ---\n")
        f.write(f"Cross folds average accuracy for {epochs} epochs : {rnd(np.mean(acc_per_fold))} and standart deviation = {rnd(np.std(acc_per_fold))}\n")
        f.write(f"Average precision : {rnd(np.mean(precision_scores))} and standart deviation = {rnd(np.std(precision_scores))}\n")
        f.write(f"Average recall : {rnd(np.mean(recall_scores))} and standart deviation = {rnd(np.std(recall_scores))}\n")
        f.write(f"Average f1 : {rnd(np.mean(f1_scores))} and standart deviation = {rnd(np.std(f1_scores))}\n")
        f.write("--- End ---")

if not os.path.isfile("generated_sentences.csv"):
    output_sentence_file()

EMBEDDING_SIZE = 100

train(
    "generated_sentences.csv",
    epochs=10,
    batch_size=128,
    folds=10,
    embedding_size=EMBEDDING_SIZE
)
