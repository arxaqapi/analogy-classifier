from google_sentence_gen import extend_embedd_generated_sentences

import random
from datetime import datetime

import numpy as np
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


def train(dataset, epochs=10, batch_size=32, folds=10, embedding_size=50):
    embedded_dataset, Y = dataset

    # embedd the dataset
    print(
        f"[Log] - Parameters : epochs = {epochs} | folds = {folds} | Word vector size = {embedding_size}")
    random.seed()

    # KFold init
    kf = KFold(n_splits=folds, shuffle=True, random_state=5)

    # Prepare data for convolutional layer
    print(f"Shape = {embedded_dataset.shape}")
    embedded_dataset = np.reshape(
        embedded_dataset,
        (embedded_dataset.shape[0], embedding_size, 4, 1)
    )

    # Parameters
    input_shape = embedded_dataset[0].shape
    fold = 1
    verbosity = 1

    print("---- START ----")
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
        )
        # Metrics
        # scores =
        cnn.evaluate(
            X_test,
            y_test,
            verbose=verbosity
        )
        save(cnn, f"fold_{fold}")
        fold += 1


EMBEDDING_SIZE = 100

train(
    extend_embedd_generated_sentences(
        "generated_sentences.csv",
        embedding_size=EMBEDDING_SIZE
    ),
    epochs=10,
    batch_size=128,
    folds=10,
    embedding_size=EMBEDDING_SIZE
)
