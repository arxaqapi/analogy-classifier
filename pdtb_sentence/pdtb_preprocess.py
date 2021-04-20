import pandas as pd
import os
import glob
import csv


def extract_columns_to_csv(data, columns, name="extract.csv", mode='a'):
    data.to_csv(name, sep='|', columns=columns, index=False, mode=mode)


def create_single_csv_from_pdtb(path="pdtb_sentences.csv", columns=[0, 24, 34]):
    """Creates a single .csv containing all pair sentences in the pdtb database

    Args:
        path (str, optional): Path to the file being created. Defaults to "pdtb_sentences.csv".
    """
    if os.path.exists(path):
        os.remove(path)
    file_list = glob.glob("pdtb/*/*.pipe")
    for f_path in file_list:
        # read as csv
        data = pd.read_csv(
            f_path,
            sep='|',
            header=None,
            dtype='string',
            error_bad_lines=False,
            engine='python'
        )
        # extract colmns to csv
        extract_columns_to_csv(data, columns=columns, name=path, mode='a')


def split_single_csv_into_relation_type_files(path):
    data_dict = {}
    with open(path, 'r') as f:
        csv_file = csv.reader(f, delimiter='|')
        for row in csv_file:
            if row != ['0', '24', '34']:
                relation = row[0]
                # row[1], row[2], row[2]
                if relation not in data_dict.keys():
                    data_dict[relation] = []
                else:
                    data_dict[relation].append(row[2:])

    for section in data_dict.keys():
        path_to_file = "pdtb_split/pdtb" + str(section) + ".pipe.csv"
        with open(path_to_file, 'w') as f:
            csv_file = csv.writer(f, delimiter='|')
            for row in data_dict[section]:
                csv_file.writerow(row)
    return data_dict


def split_single_csv_into_L1_relations(path, single_file=False):
    data_dict = {}
    prefix = "pdtb_L1_split/"
    if not os.path.exists(prefix):
        os.mkdir(prefix)
    else:
        os.system('rm -rf ' + prefix)
        os.mkdir(prefix)
    with open(path, 'r') as f:
        csv_file = csv.reader(f, delimiter='|')
        for row in csv_file:
            if row != ['11', '12', '24', '34']:
                relationA = row[0].split(".")[0]
                if relationA not in data_dict.keys():
                    data_dict[relationA] = []
                # add to dict
                data_dict[relationA].append(row[2:])

                relationB = row[1].split(".")[0]
                if relationB not in data_dict.keys():
                    data_dict[relationB] = []
                # add to dict
                data_dict[relationB].append(row[2:])
    if single_file:
        path_to_file = prefix + "pdtb_L1_relation.pipe.csv"
        with open(path_to_file, 'w') as f:
            csv_file = csv.writer(f, delimiter='|')
            for section in data_dict.keys():
                if section == '':
                    continue
                for row in data_dict[section]:
                    csv_file.writerow([section] + row)
    else:
        for section in data_dict.keys():
            if section == '':
                continue
            path_to_file = prefix + "pdtb" + str(section) + ".pipe.csv"
            with open(path_to_file, 'w') as f:
                csv_file = csv.writer(f, delimiter='|')
                for row in data_dict[section]:
                    csv_file.writerow(row)
    return data_dict


def split_single_csv_into_L2_relations(path, single_file='False'):
    data_dict = {}
    prefix = "pdtb_L2_split/"
    if not os.path.exists(prefix):
        os.mkdir(prefix)
    else:
        os.system('rm -rf ' + prefix)
        os.mkdir(prefix)
    # read big file
    with open(path, 'r') as f:
        csv_file = csv.reader(f, delimiter='|')
        for row in csv_file:
            # skip dead lines
            if row != ['11', '12', '24', '34']:
                relations_A = row[0].split('.')
                # Skip if not L2
                if len(relations_A) < 2:
                    continue
                L2_relation_A = relations_A[0] + '.' + relations_A[1]
                # add to dict rel_A
                if L2_relation_A not in data_dict.keys():
                    data_dict[L2_relation_A] = []
                data_dict[L2_relation_A].append(row[2:])

                # relations B
                relations_B = row[1].split('.')
                # Skip if not L2
                if len(relations_B) < 2:
                    continue
                L2_relation_B = relations_B[0] + '.' + relations_B[1]
                # add to dict rel_B
                if L2_relation_B not in data_dict.keys():
                    data_dict[L2_relation_B] = []
                data_dict[L2_relation_B].append(row[2:])
    if single_file:
        path_to_file = prefix + "pdtb_L2_relation.pipe.csv"
        with open(path_to_file, 'w') as f:
            csv_file = csv.writer(f, delimiter='|')
            for section in data_dict.keys():
                if section == '':
                    continue
                for row in data_dict[section]:
                    csv_file.writerow([section] + row)
    else:
        for section in data_dict.keys():
            if section == '':
                continue
                # create single files
            path_to_file = prefix + "pdtb" + str(section) + ".pipe.csv"
            with open(path_to_file, 'w') as f:
                csv_file = csv.writer(f, delimiter='|')
                for row in data_dict[section]:
                    csv_file.writerow(row)
    return data_dict


def split_single_csv_into_L3_relations(path, single_file='False'):
    data_dict = {}
    prefix = "pdtb_L3_split/"
    if not os.path.exists(prefix):
        os.mkdir(prefix)
    else:
        os.system('rm -rf ' + prefix)
        os.mkdir(prefix)
    # read big file
    with open(path, 'r') as f:
        csv_file = csv.reader(f, delimiter='|')
        for row in csv_file:
            # skip dead lines
            if row != ['11', '12', '24', '34']:
                relations_A = row[0].split('.')
                # Skip if not L3
                if len(relations_A) < 3:
                    continue
                L2_relation_A = relations_A[0] + '.' + \
                    relations_A[1] + '.' + relations_A[2]
                # add to dict rel_A
                if L2_relation_A not in data_dict.keys():
                    data_dict[L2_relation_A] = []
                data_dict[L2_relation_A].append(row[2:])

                # relations B
                relations_B = row[1].split('.')
                # Skip if not L3
                if len(relations_B) < 3:
                    continue
                L2_relation_B = relations_B[0] + '.' + relations_B[1] + '.' + relations_B[2]
                # add to dict rel_B
                if L2_relation_B not in data_dict.keys():
                    data_dict[L2_relation_B] = []
                data_dict[L2_relation_B].append(row[2:])
    if single_file:
        path_to_file = prefix + "pdtb_L3_relation.pipe.csv"
        with open(path_to_file, 'w') as f:
            csv_file = csv.writer(f, delimiter='|')
            for section in data_dict.keys():
                if section == '':
                    continue
                for row in data_dict[section]:
                    csv_file.writerow([section] + row)
    else:
        for section in data_dict.keys():
            if section == '':
                continue
                # create single files
            path_to_file = prefix + "pdtb" + str(section) + ".pipe.csv"
            with open(path_to_file, 'w') as f:
                csv_file = csv.writer(f, delimiter='|')
                for row in data_dict[section]:
                    csv_file.writerow(row)
    return data_dict


# split_single_csv_into_L1_relations("pdtb/semantic_sentence_database.csv", single_file=True)
# split_single_csv_into_L2_relations("pdtb/semantic_sentence_database.csv", single_file=True)
# split_single_csv_into_L3_relations("pdtb/semantic_sentence_database.csv", single_file=True)
