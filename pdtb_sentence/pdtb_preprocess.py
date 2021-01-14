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

def split_single_csv_into_semantic_relation_files(path):
    data_dict = {}
    prefix = "pdtb_semantic_split/"
    with open(path, 'r') as f:
        csv_file = csv.reader(f, delimiter='|')
        for row in csv_file:
            if row != ['11', '12', '24', '34']:
                relationA = row[0].split(".")[0]
                if relationA not in data_dict.keys():
                    data_dict[relationA] = []
                else:
                    data_dict[relationA].append(row[2:])
                    
                relationB = row[1].split(".")[0]
                if relationB not in data_dict.keys():
                    data_dict[relationB] = []
                else:
                    data_dict[relationB].append(row[2:])
    if os.path.exists(prefix + "all_sentences.csv"):
        os.remove(prefix + "all_sentences.csv")
    for section in data_dict.keys():
        if section == '':
            continue
        path_to_file = prefix + "pdtb" + str(section) + ".pipe.csv"
        with open(path_to_file, 'w') as f:
            csv_file = csv.writer(f, delimiter='|')
            for row in data_dict[section]:
                csv_file.writerow(row)
    return data_dict
