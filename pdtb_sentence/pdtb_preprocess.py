import pandas as pd
import os
import glob
import csv

def extract_columns_to_csv(data, columns=[0, 24, 34], name="extract.csv", mode='a'):
    data.to_csv(name, columns=columns, mode=mode)

def create_single_csv_from_pdtb(path="pdtb_sentences.csv"):
    """Creates a single .csv containing all pair sentences in the pdtb database

    Args:
        path (str, optional): Path to the file being created. Defaults to "pdtb_sentences.csv".
    """
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
        extract_columns_to_csv(data, name=path, mode='a')

def split_single_csv_into_relation_type_files(path):
    data_dict = {}
    with open(path, 'r') as f:
        csv_file = csv.reader(f, delimiter=',')
        for row in csv_file:
            if row != ['', '0', '24', '34']:
                # row[1], row[2], row[2]
                if row[1] not in data_dict.keys():
                    data_dict[row[1]] = []
                else:
                    data_dict[row[1]].append(row[2:])

    for section in data_dict.keys():
        path_to_file = "pdtb_split/pdtb" + str(section) + ".pipe.csv"
        with open(path_to_file, 'w') as f:
            csv_file = csv.writer(f, delimiter='|')
            for row in data_dict[section]:
                csv_file.writerow(row)
    return data_dict
