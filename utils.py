import csv
import numpy as np

def abcd_valid_extended(row):
    a = row[0]
    b = row[1]
    c = row[2]
    d = row[3]
    return [
        row,
        [a, c, b, d],
        [c, d, a, b],
        [c, a, d, b],
        [d, b, c, a],
        [d, c, b, a],
        [b, a, d, c],
        [b, d, a, c]
    ]

def bacd_invalid_extended(row):
    a = row[0]
    b = row[1]
    c = row[2]
    d = row[3]
    return abcd_valid_extended([b, a, c, d])

def cbad_invalid_extended(row):
    a = row[0]
    b = row[1]
    c = row[2]
    d = row[3]
    return abcd_valid_extended([c, b, a, d])

def concat(eq_list, valid=1):
    for eq in eq_list:
        eq.append(valid)

def extendGoogleDataset(path):
    """
    - open the selected dataset (here the google dataset)
    - reads it and extends the data 
    - put the data in 2 variables, (X_final, y)
    path = path to the dataset
    """
    X_final = []
    y = []
    with open(path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        line_count = 0
        X = []
        f_name = []
        for row in csv_reader:
            if row[0] == ":":
                f_name.append(row[1])
            else:
                X.append(row)
    with open("data/embedded_words.csv", mode='w') as output_file:
            writer = csv.writer(output_file, delimiter=' ')
            for equation in X:
                # equation is a list of words
                abcd = abcd_valid_extended(equation)
                X_final.extend(abcd)

                bacd = bacd_invalid_extended(equation)
                X_final.extend(bacd)
                
                cbad = cbad_invalid_extended(equation)
                X_final.extend(cbad)

                writer.writerows(abcd)
                y.extend([[1]] * 8)
                writer.writerows(bacd)
                y.extend([[0]] * 8)
                writer.writerows(cbad)
                y.extend([[0]] * 8)
    with open("data/y_embedded_words.csv", mode='w') as output_file:
        write = csv.writer(output_file, delimiter=' ')
        for row in y:
            write.writerow(row)
    return (np.array(X_final), np.array(y))

def glove_dict(embedding_size):
    """Return the dictionnary containing each word vector

    Args:
        embedding_size (int): the size of the word vectors = 50, 100, 200 or 300

    Returns:
        dict: the dictionnar containing all word vectors of size embedding_size
    """
    embeddings_dict = {}
    with open("data/glove.6B/glove.6B." + str(embedding_size) + "d.txt", 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vec = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vec
    return embeddings_dict

def embedd_dataset(dataset, embeddings_dict):
    """
        The dataset is in the form:
            X is a list of rows
            with each row being a list of words
    """
    embedded_dataset = []
    for row in dataset:
        embedded_row = []
        for word in row:
            embedded_row.append(embeddings_dict[word.lower()])
        embedded_dataset.append(embedded_row)
    return np.array(embedded_dataset)