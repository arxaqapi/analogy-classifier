from datetime import datetime
import numpy as np
import os

def save(model, name):
    today = datetime.now()
    model.save(
        "cnn_model/" +
        str(today.strftime("%d_%m_%Y__%H_%M_%S_")) +
        name +
        '.keras')


def rnd(n):
    return np.around(n, 3)


def write_to_file(log_file, text, wipe=False):
    if wipe and os.path.exists(log_file):
        os.remove(log_file)
    with open(log_file, 'a+') as f:
        f.write(text + "\n")