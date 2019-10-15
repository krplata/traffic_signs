import os
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter


def plt_count_of_classes(images_path, title):
    if os.path.exists(images_path):
        data = []
        for dirname in os.listdir(images_path):
            file_count = sum([len(f) for r, d, f in os.walk(
                os.path.join(images_path, dirname))])
            data.append((file_count, int(dirname)))

        data = sorted(data, key=itemgetter(1))
        file_count_list = []
        names = []
        for it in data:
            file_count_list.append(it[0])
            names.append(it[1])

        plt.bar(np.arange(len(names)), file_count_list,
                align='center')
        plt.xticks(np.arange(len(names)), names)
        plt.title(title)
        plt.show()


plt_count_of_classes('./data/train',
                     'Amount of samples in each class')
