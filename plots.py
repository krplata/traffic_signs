import os
import matplotlib.pyplot as plt
import numpy as np


def plt_count_of_classes(images_path, title):
    if os.path.exists(images_path):
        file_count_list = []
        for dirname in os.listdir(images_path):
            file_count = sum([len(f) for r, d, f in os.walk(
                os.path.join(images_path, dirname))])
            file_count_list.append(file_count)

        names = list(range(0, 43))
        plt.bar(np.arange(len(names)), file_count_list,
                align='center')
        plt.xticks(np.arange(len(names)), names)
        plt.title(title)
        plt.show()


plt_count_of_classes('./images', 'Amount of examples within each class')
