from extend_sentences import extend_embedd_subset
from embedd_sentences import glove_dict, load_vectors_fasttext, embedd_row, EmbeddingError
from gen_sentence_db import generate_random_aaaa_abab
from pdtb_preprocess import split_single_csv_into_semantic_relation_files
from utils import rnd, write_to_file

import os
import csv
import numpy as np
import pandas as pd

from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, BatchNormalization, Flatten, Dense, Activation
from sklearn.metrics import accuracy_score


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
    print(f"\n === X shape -> {X.shape} ===\n")
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



def create_evaluation_dataset(n):
    datadict = split_single_csv_into_semantic_relation_files("pdtb/semantic_sentence_database.csv")
    return generate_random_aaaa_abab(datadict=datadict, n=n)    


def load_and_predict(model_name, embedding_size, path_test_aaaa_dataset, path_test_abab_dataset, log_file=f"evaluation.log"):
    write_to_file(log_file, "Model name = " + model_name, wipe=True)
    write_to_file(log_file, f"[Log - evaluation.py] - Started evaluation process")
    # print(f"[Log - evaluation.py] - Started evaluation process")
    cnn = load_model("cnn_final_models/" + model_name)
    
    X = []
    y_ground_truth = []
    skipped_quadruples = 0

    word_embedding_used = 'glove'
    sentence_embedding_method = 'AVG'
    k = 1

    embedding_dict = glove_dict(embedding_size)
    # A:A::A:A
    # We embedd each row and and it to the entry array X of our model
    with open(path_test_aaaa_dataset, 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f, delimiter='|')
        for row in csv_reader:
            try:
                embedded_row = embedd_row(
                    row=row,
                    word_embedding_used=word_embedding_used,
                    sentence_embedding_method=sentence_embedding_method,
                    embedding_dict=embedding_dict,
                    embedding_size=embedding_size,
                    k=k
                )
            except EmbeddingError:
                skipped_quadruples += 1
            else:
                X.append(embedded_row)
                y_ground_truth.append([1])
    # the input of our model is reshaped to match the cnn's input requirements
    X = np.array(X)
    X = np.reshape(X, (X.shape[0], embedding_size, 4, 1))
    write_to_file(log_file, f"[Log - evaluation.py] - aaaa X shape = {X.shape}")

    y_predicted = cnn.predict(X)
    # output is a continuous value between 0 and 1
    y_predicted = [0 if val < 0.5 else 1 for val in y_predicted]
    acc = accuracy_score(y_ground_truth, y_predicted)

    write_to_file(log_file, f"[Log - evaluation.py] - aaaa score = {acc} | {rnd(acc * 100)}%")    
    
    # Now for the ABAB dataset, we practically do the same than for AAAA
    # A:B::A:B
    X = []
    y_ground_truth = []
    skipped_quadruples = 0
    # We embedd each row and and it to the entry array X of our model
    with open(path_test_abab_dataset, 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f, delimiter='|')
        for row in csv_reader:
            try:
                embedded_row = embedd_row(
                    row=row,
                    word_embedding_used=word_embedding_used,
                    sentence_embedding_method=sentence_embedding_method,
                    embedding_dict=embedding_dict,
                    embedding_size=embedding_size,
                    k=k
                )
            except EmbeddingError:
                skipped_quadruples += 1
            else:
                X.append(embedded_row)
                y_ground_truth.append([1])
    X = np.array(X)
    X = np.reshape(X, (X.shape[0], embedding_size, 4, 1))
    write_to_file(log_file, f"[Log - evaluation.py] - abab X shape = {X.shape}")

    y_predicted = cnn.predict(X)
    # output is a continuous value between 0 and 1
    y_predicted = [0 if val < 0.5 else 1 for val in y_predicted]
    acc = accuracy_score(y_ground_truth, y_predicted)
    
    acc = accuracy_score(y_ground_truth, y_predicted)
    write_to_file(log_file, f"[Log - evaluation.py] - abab score = {acc} | {rnd(acc * 100)}%")
    # f1 score / precision / recall needs to be calculated


WE_USED = 'glove' # 'fasttext' or 'glove'
EMBEDDING_SIZE = 50
K = 1
SE_USED = 'AVG' # 'AVG' or 'DCT'
MODEL_NAME = f"final_model_{WE_USED}_e{EMBEDDING_SIZE}_{SE_USED}"

if not os.path.exists(f"cnn_final_models/{MODEL_NAME}"):
    train_on_full_dataset(
        "semantic_sentence_database.csv",
        epochs=10,
        batch_size=128,
        embedding_size= 300 if WE_USED == 'fasttext' else EMBEDDING_SIZE,
        word_embedding_used=WE_USED,
        sentence_embedding_method=SE_USED,
        k=K
    )

create_evaluation_dataset(n=20000)

load_and_predict(
    MODEL_NAME,
    EMBEDDING_SIZE,
    "evaluation_files/20000_generated_aaaa_sentences.csv",
    "evaluation_files/20000_generated_abab_sentences.csv",
    log_file=f"evaluation_{SE_USED}_e{EMBEDDING_SIZE}.log"
)