import csv
import random


def randomly_generate_n_sentence_quadruples(data_dict, n_sentences=20000):
    """
    For the moment we only work with 'Explicit' R relation
    Randomly select 2 sentences sharing the same relation and making it into a quadruple
    """
    print("[Log] - Randomly generating the sentence quadruples...")
    sent_pair_list = data_dict['Explicit']
    with open("explicit_sentence_database.csv", 'w') as f:
        csv_file = csv.writer(f, delimiter='|')
        for _ in range(n_sentences):
            quadruple = random.choice(
                sent_pair_list) + random.choice(sent_pair_list)
            csv_file.writerow(quadruple)


def generate_random_selected_quadruples(datadict, n=20000):
    with open("semantic_sentence_database.csv", 'w') as f:
        csv_file = csv.writer(f, delimiter='|')
        for relation in datadict.keys():
            if relation == '':
                continue
            rel_sentences = datadict[relation]
            for _ in range(int(n / 4)):
                ab = random.choice(rel_sentences)
                cd = random.choice(rel_sentences)
                quad = (ab+cd)
                csv_file.writerow(quad)
