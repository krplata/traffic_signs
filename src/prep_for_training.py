import Augmentor
from Augmentor import Pipeline as augpipe
import cv2
import argparse
import random
import os
import sys
import fnmatch
import glob
import shutil
import pre_run


def to_jpg(src_path, output_dir=None):
    '''
    Converts an image at src_path to *.jpg file.
    If output_dir isn't specified, the output image will 
    be placed at dirname(src_path).

    Parameters:
        src_path   (str): Path to an image that can be opened by cv2, 
                          with cv2.IMREAD_COLOR flag.
        output_dir (str): Path to output directory. 
    '''
    image = cv2.imread(src_path, cv2.IMREAD_COLOR)
    no_extension = os.path.splitext(src_path)[0]
    converted = no_extension + '.jpg'
    image = pre_run.im_prepare(image)
    if not output_dir:
        cv2.imwrite(converted, image)
    else:
        filename = os.path.basename(converted)
        cv2.imwrite(os.path.join(output_dir, filename), image)


def ppm_dir_to_jpg(source_dir, output_dir):
    '''
    Converts all *.ppm files within a directory to *.jpg.
    Output directory will be created, if it doesn't already exist.

    Parameters:
        source_dir (str): Path to the directory with source (*.ppm) files.
        output_dir (str): Path to the output directory.
    '''
    print(f"---> Converting .ppm files from {source_dir}\n")
    for r, d, f in os.walk(source_dir):
        for file in os.listdir(r):
            if fnmatch.fnmatch(file, '*.ppm'):
                output_path = r.replace(source_dir, output_dir)
                os.makedirs(output_path, exist_ok=True)
                to_jpg(os.path.join(r, file), output_path)


def count_files(dirpath, extension='*', recursive=False):
    '''
    Returns the number of files within a directory tree.

    Parameters:
        dirpath (str): Path to the directory, in which the files will be counted.
        extension (str): Extension of the files that will be considered.
        recursive (bool): Count files in subdirectories.
    '''
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
    geometric_aug = [augpipe.skew, augpipe.rotate_random_90,
                     augpipe.flip_random]
    color_aug = [augpipe.random_brightness, augpipe.random_contrast]

    for directory in os.listdir(source_dir):
        images_dir = os.path.join(source_dir, directory)
        file_count = count_files(images_dir)
        enhancement_factor = int(dest_class_size/file_count)
        for it in range(0, enhancement_factor):
            p = augpipe(source_directory=images_dir,
                        output_directory='./output')
            rand_geo = geometric_aug[random.randint(0, len(geometric_aug) - 1)]
            rand_color = color_aug[random.randint(0, len(color_aug) - 1)]
            rand_geo(p, probability=1)
            rand_color(p, probability=0.5, min_factor=0.6, max_factor=0.9)
            p.greyscale(1.0)
            p.process()
        output_path = os.path.join(images_dir, "output")
        for f in os.listdir(output_path):
            shutil.move(os.path.join(output_path, f), images_dir)
        os.rmdir(output_path)


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
