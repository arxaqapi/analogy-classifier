import utils

import random
from datetime import datetime

import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Flatten, Dense, Activation
from sklearn.model_selection import KFold


def cnn_model(shape=(50, 4, 1)):
    model = Sequential([
        Conv2D(filters=128, kernel_size=(1, 2),
               strides=(1, 2), input_shape=shape),
        BatchNormalization(axis=-1),
        Activation('relu'),
        Conv2D(filters=64, kernel_size=(2, 2), strides=(2, 2)),
        BatchNormalization(axis=-1),
        Activation('relu'),
        Flatten(),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', metrics=[
                  'accuracy'], optimizer='adam')
    return model


def save(model, name):
    today = datetime.now()
    model.save("cnn_model/" +
               str(today.strftime("%d_%m_%Y__%H_%M_%S")) + name + '.h5')


def train(dataset, epochs=10, folds=10, embedding_size=50):
    """Train the convolutional neural net and outputs the accuracy

    Args:
        dataset (tuple): tuple of the forw (x, y) with x beeing the input data and y the awaited results
        epochs (int, optional): Number of epochs for the neural net to train. Defaults to 10.
        folds (int, optional): Number of folds for the KFold model selection. Defaults to 10.
        embedding_size (int, optional): Size of the word vectors, can be one the following 50, 100, 200, 300. Defaults to 50.
    """

    X, Y = dataset
    
    # embedd the dataset
    print("[Log] - Embedding starts")
    embedded_dataset = utils.embedd_dataset(X, utils.glove_dict(embedding_size))
    print(f"[Log] - Parameters : epochs = {epochs} | folds = {folds} | Word vector size = {embedding_size}")
    random.seed()
    kf = KFold(n_splits=folds, shuffle=True, random_state=5)
    embedded_dataset = embedded_dataset.reshape(len(X), embedding_size, 4, 1)

    # Parameters
    input_shape = embedded_dataset[0].shape
    batch_size = int(len(embedded_dataset) / 10)
    fold = 1
    verbosity = 0

    # Define per-fold score containers
    acc_per_fold = []
    loss_per_fold = []

    print("---- START ----")
    for train_index, test_index in kf.split(embedded_dataset):
        
        X_train = embedded_dataset[train_index]
        y_train = Y[train_index]
        
        X_test = embedded_dataset[test_index]
        y_test = Y[test_index]
        
        cnn = cnn_model(input_shape)
        
        history = cnn.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbosity
        )
        # Metrics
        scores = cnn.evaluate(
            X_test,
            y_test,
            verbose=verbosity
        )
        print(
            f'Score per fold n°{fold}: {cnn.metrics_names[0]} of {scores[0]}; {cnn.metrics_names[1]} of {scores[1]*100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])
        save(cnn, f"fold_{fold}")
        fold += 1

    print('------------------------------------------------------------------------')
    print('Score per fold')
    for i in range(0, len(acc_per_fold)):
        print('------------------------------------------------------------------------')
        print(
            f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
    print(f'> Loss: {np.mean(loss_per_fold)}')
    print('------------------------------------------------------------------------')


train(utils.extendGoogleDataset("data/google/questions-words.txt"), epochs=10, folds=10, embedding_size=50)
