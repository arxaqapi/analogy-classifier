from extend_sentences import extend_embedd_subset
from embedd_sentences import glove_dict, load_vectors_fasttext

import numpy as np
import pandas as pd

from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, BatchNormalization, Flatten, Dense, Activation


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


def save_model(model, name="test_final_model"):
    model.save("cnn_final_models/" + name)


def train_on_full_dataset(
        dataset_path,
        word_embedding_used,
        sentence_embedding_method,
        k,
        epochs=10,
        batch_size=32,
        embedding_size=50):

    if word_embedding_used == 'glove':
        embedding_dict = glove_dict(embedding_size)
    elif word_embedding_used == 'fasttext':
        embedding_dict = load_vectors_fasttext()
    else:
        raise ValueError(
            "word_embedding_used should be 'glove' or 'fasttext' in extend_embedd_sentences()")
    if sentence_embedding_method == 'DCT':
        # embedding_size = 300
        embedding_size *= k
    # Parameters
    input_shape = (embedding_size, 4, 1)
    verbosity = 1

    print("[Log] ---- Start training the final model ----")

    X, y = extend_embedd_subset(
        dataset=dataset_path,
        word_embedding_used=word_embedding_used,
        sentence_embedding_method=sentence_embedding_method,
        embedding_dict=embedding_dict,
        embedding_size=embedding_size,
        k=k
    )
    print("[Log] - Extension of the dataset finished, training starts ...")
    cnn = cnn_model(input_shape)

    cnn.fit(
        X,
        y,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbosity
    )
    save_model(cnn, MODEL_NAME)



def create_evaluation_dataset():
    return


def load_and_predict(model_name):
    cnn = load_model(model_name)
    # model.predict()
    # await 100% accuracy
    # confusion matrix
    # precision
    # recall
    # f1-score
    # accuracy
    return


WE_USED = 'glove' # 'fasttext' or 'glove'
EMBEDDING_SIZE = 50
K = 1
SE_USED = 'AVG' # 'AVG' or 'DCT'
MODEL_NAME = f"final_model_{WE_USED}_e{EMBEDDING_SIZE}_{SE_USED}"

train_on_full_dataset(
    "semantic_sentence_database.csv",
    epochs=10,
    batch_size=128,
    embedding_size= 300 if WE_USED == 'fasttext' else EMBEDDING_SIZE,
    word_embedding_used=WE_USED,
    sentence_embedding_method=SE_USED,
    k=K
)