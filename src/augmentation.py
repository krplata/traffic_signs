import Augmentor
from Augmentor import Pipeline as augpipe
import cv2
import random
import os
import sys
from fnmatch import fnmatch
import shutil
from . import reshape
from . import dataset_utils as utils


def to_grey_jpg(src_path, output_dir=None):
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
    image = reshape.im_prepare(image)
    if not output_dir:
        cv2.imwrite(converted, image)
    else:
        filename = os.path.basename(converted)
        cv2.imwrite(os.path.join(output_dir, filename), grey)


def ppm_dir_to_grey_jpg(source_dir, output_dir):
    '''
    Converts all *.ppm files within a directory to greyscale *.jpg.
    Output directory will be created, if it doesn't already exist.

    Parameters:
        source_dir (str): Path to the directory with source (*.ppm) files.
        output_dir (str): Path to the output directory.
    '''
    print(f"-- Converting .ppm files from {source_dir}\n")
    for r, d, f in os.walk(source_dir):
        for file in os.listdir(r):
            if fnmatch(file, '*.ppm'):
                to_grey_jpg(os.path.join(r, file))
                os.remove(os.path.join(r, file))


def generate_augmented(source_dir, dest_class_size):
    geometric_aug = [augpipe.skew, augpipe.rotate_random_90,
                     augpipe.flip_random]
    color_aug = [augpipe.random_brightness, augpipe.random_color]

    for directory in os.listdir(source_dir):
        images_dir = os.path.join(source_dir, directory)
        file_count = utils.count_files(images_dir)
        enhancement_factor = int(dest_class_size/file_count)
        for it in range(0, enhancement_factor):
            p = augpipe(source_directory=images_dir,
                        output_directory='./output')
            rand_geo = geometric_aug[random.randint(0, len(geometric_aug) - 1)]
            rand_color = color_aug[random.randint(0, len(color_aug) - 1)]
            rand_geo(p, probability=1)
            rand_color(p, probability=0.5, min_factor=0.6, max_factor=0.9)
            p.process()
        output_path = os.path.join(images_dir, "output")
        for f in os.listdir(output_path):
            shutil.move(os.path.join(output_path, f), images_dir)
        os.rmdir(output_path)
