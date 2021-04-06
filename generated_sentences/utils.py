from datetime import datetime
import numpy as np


def save(model, name):
    today = datetime.now()
    model.save(
        "cnn_model/" +
        str(today.strftime("%d_%m_%Y__%H_%M_%S_")) +
        name +
        '.keras')


def rnd(n):
    return np.around(n, 3)
