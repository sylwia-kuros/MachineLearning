import csv

from typing import Dict


def get_vocabulary_dict() -> Dict[int, str]:
    """Read the fixed vocabulary list from the datafile and return.

    :return: a dictionary of words mapped to their indexes
    """

    # Parse data from the 'data/vocab.txt' file
    dictionary = {}

    with open('data/vocab.txt', newline='') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            dictionary[int(row[0])] = row[1]
    file.close()

    return dictionary
