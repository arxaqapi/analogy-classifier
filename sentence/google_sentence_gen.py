import csv
import numpy as np

from sentence_embedding import glove_dict
from sentence_extend import abcd_valid_extended, bacd_invalid_extended, cbad_invalid_extended


def split_google_dataset(path, to):
    """Split the google word analogy dataset into separate files,
    one for each category

    Args:
        path (string): path the google ```questions-words.txt``` dataset

    Returns:
        dict: the dictionnary containing the data splited into categories
    """
    dataset = {}
    current_section_name = ''
    with open(path, 'r') as f:
        csv_reader = csv.reader(f, delimiter=' ')
        for row in csv_reader:
            if row[0] == ':':
                # new section
                current_section_name = row[1]
                dataset[row[1]] = []
            else:
                dataset[current_section_name].append(row)

    for section in dataset.keys():
        with open(to + section + '.txt', 'w') as f:
            csv_writer = csv.writer(f, delimiter=' ')
            for row in dataset[section]:
                csv_writer.writerow(row)
    return dataset


def replace_with(sentence, car, word):
    return sentence.replace(car, word)


# Common capital cities template sentences
ccc_sentences = (
    ("They traveled to $", "They took a trip to $"),
    ("She arrived yesterday in $", "She just landed in $"),
    ("We just came back from $", "We arrived yesterday from $"),
    ("I took my vacations in $", "I flew to $ for these summer vacation"),
    (
        "My internship supervisor is born in $",
        "My internship supervisor comes from $"
    )
)


def gen_ccc_sentences(path, sentences, output_file="temp/ccc_sentences.csv", write_mode='w'):
    """Generate len(sentences)*4*nb_rows new sentences based on the sentences template

    Args:
        path (string): path to the input word file
        sentences (list): the templates sentences

    Returns:
        list: the newly generated sentences
    ```
    - s1a : s2b :: s1c : s2d
    - s2a : s1b :: s2c : s1d
    - s1a : s1b :: s1c : s1d
    - s2a : s2b :: s2c : s2d
    ```
    """
    generated_sentences = []
    with open(path, 'r') as f:
        csv_file = csv.reader(f, delimiter=' ')
        for row in csv_file:
            # s1a : s2b :: s1c : s2d
            # s2a : s1b :: s2c : s1d
            for start_index in range(2):
                for sentence_tuple in sentences:
                    sentence_row = []
                    i = start_index
                    for word in row:
                        sentence_row.append(replace_with(
                            sentence_tuple[i % 2], '$', word.strip())
                        )
                        i += 1
                    generated_sentences.append(sentence_row)
            # s1a : s1b :: s1c : s1d
            # s2a : s2b :: s2c : s2d
            for sentence_tuple in sentences:
                for sent in sentence_tuple:
                    sentence_row = []
                    for word in row:
                        sentence_row.append(
                            replace_with(sent, '$', word.strip())
                        )
                    generated_sentences.append(sentence_row)
    # Write to file
    with open(output_file, write_mode) as f:
        csv_writer = csv.writer(f, delimiter=";")
        for row in generated_sentences:
            csv_writer.writerow(row)

    return generated_sentences


# gram6-nationality-adjective template sentences
gna_sentences = (
    ("My friend comes from $", "I have a $ friend"),
    ("The culture in $ is very rich", "The $ culture is very rich"),
)


def gen_gna_sentences(path, sentences, output_file="gna_sentences.csv", write_mode='w'):
    """Generate len(sentences)*1*nb_rows new sentences based on the sentences template

    Args:
        path (string): path to the input word file
        sentences (list): the templates sentences

    Returns:
        list: the newly generated sentences
    ```
    - s1a : s2b :: s1c : s2d
    ```
    """
    generated_sentences = []
    with open(path, 'r') as f:
        csv_file = csv.reader(f, delimiter=' ')
        for row in csv_file:
            # s1a : s2b :: s1c : s2d
            for sentence_tuple in sentences:
                sentence_row = []
                i = 0
                for word in row:
                    sentence_row.append(
                        replace_with(
                            sentence_tuple[i % 2],
                            '$',
                            word.strip()
                        ))
                    i += 1
                generated_sentences.append(sentence_row)
    # Write to file
    with open(output_file, write_mode) as f:
        csv_writer = csv.writer(f, delimiter=";")
        for row in generated_sentences:
            csv_writer.writerow(row)

    return generated_sentences


def extend_embedd_generated_sentences(path, embedding_size=100):
    """Extend and embedds the sentence dataset !!

    Args:
        path (string): path to the sentences.csv file containing the sentences

    Returns:
        tuple: X,y arrays containing the embedded dataset
    """
    # Split the dataset into temp/ directory
    split_google_dataset("../data/google/questions-words.txt", "temp/")
    
    # Generate the new sentences forming the dataset
    gen_ccc_sentences(
        "temp/capital-common-countries.txt",
        ccc_sentences,
        output_file='sentences.csv',
        write_mode='w'
    )
    gen_gna_sentences(
        "temp/gram6-nationality-adjective.txt",
        gna_sentences,
        output_file='sentences.csv',
        write_mode='a'
    )

    # extend then embedd the dataset
    embedding_dict = glove_dict(embedding_size, "../data/glove.6B/")
    # Create X, Y placeholders
    X = []
    y = []
    with open(path, 'r') as f:
        csv_reader = csv.reader(f, delimiter=';')
        for row in csv_reader:
            # extend valid
            abcd = abcd_valid_extended(
                row,
                embedding_dict=embedding_dict,
                embedding_size=embedding_size
            )
            X.extend(abcd)
            y.extend([[1]] * 8)
            # extend invalid
            bacd = bacd_invalid_extended(
                row,
                embedding_dict=embedding_dict,
                embedding_size=embedding_size
            )
            cbad = cbad_invalid_extended(
                row,
                embedding_dict=embedding_dict,
                embedding_size=embedding_size
            )
            X.extend(bacd)
            X.extend(cbad)
            y.extend([[0]] * 8)
            y.extend([[0]] * 8)

    # return X y
    return np.array(X), np.array(y)
