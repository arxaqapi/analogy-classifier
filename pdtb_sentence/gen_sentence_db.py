import csv
import random
# preprocess word, if contains "*'*" or "'*" or "*'" split these words


# if we generate all possible quadruples -> 18426 * 18426 = 339_517_476 sentences (Way too much, let's try)

# Deprecated
def danger_generate_quadruples(data_dict):
    # get dataset
    # for for
    # to file
    print("[/!\\] - This is a slow and memory consuming function call\n[/!\\] - The generated file 'danger.csv', is bigger than 100Go")
    with open("danger.csv", 'w') as f:
        csv_file = csv.writer(f, delimiter='|')
        for sent_pair in data_dict['Explicit']:  # [::100]:
            for snd_pair in data_dict['Explicit']:
                if sent_pair != snd_pair:
                    # We do not want to overwrite sent_pair by using .extend()
                    csv_file.writerow(sent_pair + snd_pair)


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
