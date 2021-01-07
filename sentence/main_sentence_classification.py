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


def train(dataset_path, epochs=10, batch_size=32, folds=10, embedding_size=50):
    print(
        f"[Log] - Parameters : epochs = {epochs} | batch_size = {batch_size} |folds = {folds} | Word vector size = {embedding_size}")

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
        # Metrics
        cnn.evaluate(
            X_test,
            y_test,
            verbose=verbosity
        )
        save(cnn, f"fold_{fold}")
        fold += 1

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
