import re
import csv
import sys
import numpy as np
import pandas as pd

import sentence_templates as st
from sentence_embedding import glove_dict, embedd_row, EmbeddingError
from sentence_extend import abcd_valid_extended, bacd_invalid_extended, cbad_invalid_extended


SENT_FILE = "generated_sentences.csv"


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
    with open(path, 'r', encoding='utf-8') as f:
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


def fix_a_an(sent):
    obj = "[a,an]"
    splitted = sent.split(' ')
    try:
        index = splitted.index(obj)
    except:
        return sent
    else:
        if splitted[index + 1][0].lower() in ['a', 'e', 'i', 'o', 'u']:
            splitted[index] = "an"
        else:
            splitted[index] = "a"
    return ' '.join(splitted)


def fix_is_are(sent):
    obj = "[is,are]"
    splitted = sent.split(' ')
    try:
        index = splitted.index(obj)
    except:
        return sent
    else:
        if splitted[index - 1][len(splitted[index - 1]) - 1].lower() in ['s']:
            splitted[index] = "are"
        else:
            splitted[index] = "is"
    return ' '.join(splitted)


def ccc_gen():  # capital-common-cities sentences gen
    generated_sentences = []
    with open("google_split/capital-common-countries.txt", mode='r', encoding="utf-8") as f:
        csv_reader = csv.reader(f, delimiter=' ')
        for row in csv_reader:
            # s1a : s2b :: s1c : s2d
            i = 0
            for sentence_tuple in st.ccc_sentences:
                sentence_row = []
                for word in row:
                    sentence_row.append(
                        replace_with(
                            sentence_tuple[i % 2],
                            '$',
                            word.strip()
                        )
                    )
                    i += 1
                generated_sentences.append(sentence_row)
    return generated_sentences


def cis_gen():  # city-in-state sentence gen
    generated_sentences = []
    with open("google_split/city-in-state.txt", mode='r', encoding="utf-8") as f:
        csv_reader = csv.reader(f, delimiter=' ')
        for row in csv_reader:
            for sentence in st.cis_sentences:
                sentence_row = []
                for word in row:
                    sentence_row.append(
                        replace_with(
                            sentence,
                            '$',
                            word.strip()
                        )
                    )
                generated_sentences.append(sentence_row)
    return generated_sentences


def cur_gen():  # currency sent gen
    generated_sentences = []
    with open("google_split/currency.txt", mode='r', encoding="utf-8") as f:
        csv_reader = csv.reader(f, delimiter=' ')
        for row in csv_reader:
            # s1a : s2b :: s1c : s2d
            i = 0
            for sentence_tuple in st.cur_sentences:
                sentence_row = []
                for word in row:
                    sentence_row.append(
                        replace_with(
                            sentence_tuple[i % 2],
                            '$',
                            word.strip()
                        )
                    )
                    i += 1
                generated_sentences.append(sentence_row)
    return generated_sentences


def fam_gen():  # family sent gen
    generated_sentences = []
    exclusion_list = ["his", "he", "prince", "policeman"]
    with open("google_split/family.txt", mode='r', encoding="utf-8") as f:
        csv_reader = csv.reader(f, delimiter=' ')
        for row in csv_reader:
            if row[0] in exclusion_list or row[2] in exclusion_list:
                continue
            for sentence in st.fam_sentences:
                sentence_row = []
                for word in row:
                    sentence_row.append(
                        fix_is_are(
                            replace_with(
                                sentence,
                                '$',
                                word.strip()
                            )
                        )
                    )
                generated_sentences.append(sentence_row)
    return generated_sentences


def gna_gen():  # nationality adj sent gen
    generated_sentences = []
    with open("google_split/gram6-nationality-adjective.txt", mode='r', encoding="utf-8") as f:
        csv_reader = csv.reader(f, delimiter=' ')
        for row in csv_reader:
            # s1a : s2b :: s1c : s2d
            i = 0
            for sentence_tuple in st.gna_sentences:
                sentence_row = []
                for word in row:
                    sentence_row.append(
                        fix_a_an(
                            replace_with(
                                sentence_tuple[i % 2].replace(u'\xa0', ' '),
                                '$',
                                word.strip()
                            )
                        )
                    )
                    i += 1
                generated_sentences.append(sentence_row)
    return generated_sentences


def opp_dict(path="google_split/gram2-opposite.txt"):
    # 28 valeur dans dict for "google_split/gram2-opposite.txt"
    data_dict = {}
    df = pd.read_csv(path, delimiter=' ', header=None)
    step = 28
    for i in range(int(812 / step)):
        data_dict[df[0][i * step]] = df[1][i * step]
    return data_dict


def get_opposite_pair(sentence, data_dict):
    sentence_norm = sentence.split(' ')
    sentence_opp = sentence.split(' ')
    pattern = re.compile(r'^\[.+\]$')
    for w in sentence_norm:
        if pattern.search(w):
            # Found the word to replace with opposite
            index = sentence_norm.index(w)
            sentence_norm[index] = w.strip('[]')
            sentence_opp[index] = data_dict[w.strip('[]')]
            if sentence_norm[index - 1] in ["a", "an"]:
                sentence_norm[index - 1] = "[a,an]"
            if sentence_opp[index - 1] in ["a", "an"]:
                sentence_opp[index - 1] = "[a,an]"
    return list(map(fix_a_an, [' '.join(sentence_norm), ' '.join(sentence_opp)]))


def opp_gen():  # opposite sent gen
    data_dict = opp_dict()
    generated_sentences = []
    for first_sent in st.opp_sentences:
        for snd_sent in st.opp_sentences:
            if first_sent != snd_sent:
                sentence_row = []
                sentence_row.extend(
                    get_opposite_pair(first_sent, data_dict)
                )
                sentence_row.extend(
                    get_opposite_pair(snd_sent, data_dict)
                )
                generated_sentences.append(sentence_row)
    return generated_sentences


def output_sentence_file():
    split_google_dataset("../data/google/questions-words.txt", "google_split/")
    dataset = []
    dataset.extend(ccc_gen())
    dataset.extend(cis_gen())
    dataset.extend(cur_gen())
    dataset.extend(fam_gen())
    dataset.extend(gna_gen())
    dataset.extend(opp_gen())
    with open(SENT_FILE, 'w', encoding='utf-8') as f:
        csv_writer = csv.writer(f, delimiter='|')
        csv_writer.writerows(dataset)
    return dataset


def extend_embedd_subset(dataset, embedding_dict, embedding_size):
    X = []
    y = []
    skipped_quadruples = 0
    with open(dataset, 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f, delimiter='|')
        for row in csv_reader:
            try:
                embedded_row = embedd_row(
                    row,
                    embeddings_dict=embedding_dict,
                    embedding_size=embedding_size
                )
            except EmbeddingError:
                skipped_quadruples += 1
            else:
                # not executed if error
                abcd = abcd_valid_extended(
                    embedded_row
                )
                X.extend(abcd)
                y.extend([[1]] * 4) # 8 - 4
                bacd = bacd_invalid_extended(
                    embedded_row
                )
                cbad = cbad_invalid_extended(
                    embedded_row
                )
                X.extend(bacd)
                X.extend(cbad)
                y.extend([[0]] * 4) # 8 - 4
                y.extend([[0]] * 4) # 8 - 4
    X = np.array(X)
    X = np.reshape(X, (X.shape[0], embedding_size, 4, 1))
    return X, np.array(y)