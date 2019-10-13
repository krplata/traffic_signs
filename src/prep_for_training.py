import Augmentor
import cv2

import argparse

import os
import sys
import fnmatch
import glob
import pre_run
import pandas as pd

def to_jpg(src_path, output_dir=None):
    image = cv2.imread(src_path, cv2.IMREAD_COLOR)
    no_extension = os.path.splitext(src_path)[0]
    converted = no_extension + '.jpg'
    image = pre_run.im_prepare(image)
    if not output_dir:
        cv2.imwrite(converted, image)
    else:
        filename = os.path.basename(converted)
        cv2.imwrite(os.path.join(output_dir, filename), image)


# Replicates source file tree with .ppm files converted to .jpg
def ppm_dir_to_jpg(source_dir, output_dir):
    print(f"---> Converting .ppm files from {source_dir}\n")
    for r, d, f in os.walk(source_dir):
        for file in os.listdir(r):
            if fnmatch.fnmatch(file, '*.ppm'):
                output_path = r.replace(source_dir, output_dir)
                os.makedirs(output_path, exist_ok=True)
                to_jpg(os.path.join(r, file), output_path)


def count_files(dirpath, extension='*', recursive=False):
    if not recursive:
        return len(glob.glob1(dirpath, extension))
    else:
        file_counter = 0
        for r, d, f in os.walk(dirpath):
            for file in f:
                if file.endswith(extension):
                    file_counter += 1
        return file_counter


def generate_augmented(source_dir, dest_class_size):
    for directory in os.listdir(source_dir):
        images_dir = os.path.join(source_dir, directory)

        p = Augmentor.Pipeline(source_directory=images_dir,
                               output_directory='./')
        p.skew(probability=1)
        p.rotate(probability=0.7, max_left_rotation=20, max_right_rotation=20)
        p.shear(probability=0.2, max_shear_left=10, max_shear_right=10)
        p.flip_left_right(probability=0.5)
        p.flip_top_bottom(probability=0.5)

        file_count = count_files(images_dir)
        p.sample(dest_class_size - file_count)


# Replaces the names generated by Augmentor, with classId_runningID.jpg
def cleanup_names(dest_path):
    for r, d, f in os.walk(dest_path):
        running_number = 0
        for file in os.listdir(r):
            file_path = os.path.join(r, file)
            if os.path.isfile(file_path) and fnmatch.fnmatch(file, '*.jpg'):
                index_str = str(running_number).zfill(5)
                replacement = os.path.join(
                    r, f"{os.path.basename(r)}_{index_str}.jpg")
                os.rename(file_path, replacement)
                running_number += 1