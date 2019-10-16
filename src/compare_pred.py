import pandas as pd
import argparse
import os


def accuracy_on_generated(csv_file):
    source_data = pd.read_csv(csv_file, dtype=str, sep=',')
    values = source_data.values
    counter = 1
    for value in values:
        filepath, prediction = value
        if os.path.dirname(filepath) == prediction:
            counter += 1
    accuracy = "{0: .2f}".format(round(counter/len(values) * 100, 4))
    print(f"Accuracy is : {counter} / {len(values)} = {accuracy} %")
