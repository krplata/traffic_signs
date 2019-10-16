import pandas as pd
import argparse
import os


def accuracy_on_generated(csv_file):
    source_data = pd.read_csv(csv_file, dtype=str, sep=';')
    values = source_data.values
    counter = 0
    for value in values:
        filepath, prediction = value
        if os.path.dirname(filepath) == prediction:
            counter += 1
    accuracy = "{0: .2f}".format(round(counter/len(values) * 100, 4))
    print(f"Accuracy is : {counter} / {len(values)} = {accuracy} %")


def accuracy_on_external(source_csv, pred_csv):
    source_data = pd.read_csv(source_csv, dtype=str, sep=';')
    predict_data = pd.read_csv(pred_csv, dtype=str, sep=';')
    counter = 0
    values = predict_data.values
    for value in values:
        filepath, prediction = value
        pred_class_id = f"{prediction}".lstrip("0")
        row = source_data.loc[source_data['Filename'] == filepath]
        if row['ClassId'].values[0] == pred_class_id:
            counter += 1
    accuracy = "{0: .2f}".format(round(counter/len(values) * 100, 4))
    print(f"Accuracy is : {counter} / {len(values)} = {accuracy} %")
