import pandas as pd
import detection as dt
import cv2
from skimage.feature import hog
import os
from fnmatch import fnmatch
import random


def get_sign_coords(dataframe, filename):
    rows = dataframe.loc[dataframe['filename'] == filename]
    squares = []
    for index, row in rows.iterrows():
        squares.append((row['x1'], row['y1'], row['x2'], row['y2']))
    return squares


def is_within_rectangle(src, dest):
    x1_p, y1_p = src
    x1_dest, y1_dest, x2_dest, y2_dest = dest
    return x1_dest <= x1_p <= x2_dest and y1_dest <= y1_p <= y2_dest


def is_valid_negative(point, filename, dataframe):
    for box in get_sign_coords(dataframe, filename):
        if is_within_rectangle(point, box):
            return False
    return True


detection_dir = 'data/detection/dirty'
names = ['filename', 'x1', 'y1', 'x2', 'y2', 'class']
data = pd.read_csv('./data/detection/dirty/gt.txt', sep=';')
data.columns = names
index = 0
im_index = 0
print(data)
# for fname in os.listdir(detection_dir):
#     if fnmatch(fname, '*.ppm'):
#         image = cv2.imread(os.path.join(
#             detection_dir, fname), cv2.IMREAD_COLOR)
#         x, y, z = image.shape
#         pyramid = dt.Image_Pyramid(image, 1.0, (x, y))
#         generator = pyramid.sliding_window((32, 32), (32, 32))
#         for item in generator:
#             x1, y1, x2, y2 = item[1]
#             if is_valid_negative((x1, y1), fname, data) and is_valid_negative((x2, y2), fname, data):
#                 cv2.imwrite(f"data/detection/train/negative/{index}.jpg", item[0])
#                 index += 1
#         im_index += 1
#     if im_index >= 90:
#         break
