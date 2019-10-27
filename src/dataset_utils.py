import os
from fnmatch import fnmatch
import glob
import sys
import requests
from zipfile import ZipFile
from tqdm import tqdm
import shutil


def cleanup_augmentor_names(dest_path):
    '''
    Changes the *.jpg files names outputted by the Augmentor pipeline.
    All files will be renamed to the following format: classname_runningnumber.jpg.
    Function takes into consideration all directories in the dest_path tree.

    Parameters:
        dest_path (str): Path to the root of Augmentor pipeline output.
    '''
    for r, d, f in os.walk(dest_path):
        running_number = 0
        for file in os.listdir(r):
            file_path = os.path.join(r, file)
            if os.path.isfile(file_path) and fnmatch(file, '*.jpg'):
                index_str = str(running_number).zfill(5)
                replacement = os.path.join(
                    r, f"{os.path.basename(r)}_{index_str}.jpg")
                os.rename(file_path, replacement)
                running_number += 1


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


def fetch_file(url, target_dir, filename=''):
    '''
    Downloads a file from a specified url.

    Parameters:
        url (str): Link used for downloading.
        target_dir (str): Directory for storing the downloaded file.
        filename (str): If not specified, name will be determined based on the text behind the last '/' in the url.
    '''
    if not filename:
        filename = url.rsplit(sep='/', maxsplit=1)[1]
    r = requests.get(url, stream=True)
    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024
    t = tqdm(total=total_size, unit='iB', unit_scale=True, desc=filename)
    with open(os.path.join(target_dir, filename), 'wb') as f:
        for data in r.iter_content(block_size):
            t.update(len(data))
            f.write(data)


def download_files(urls, dest_dir, force=False):
    '''
    Download all files from provided links.

    Parameters:
        urls (list): List of urls used for downloading files.
        dest_dir (str): Directory used for storing the downloaded files.
        force (bool, optional): Redownload existing files. 
    '''
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)
    for url in urls:
        filename = url.rsplit(sep='/', maxsplit=1)[1]
        file_path = os.path.join(dest_dir, filename)
        if not os.path.exists(file_path) or force:
            fetch_file(url, dest_dir, filename)
        else:
            print(f"-- File {filename} already exists. Skipping download.")


def unzip_files(source_dir, target_dir):
    '''
    Extracts all *.zip files within a directory, to a target path.

    Parameters:
        source_dir (str): Path containing *.zip files.
        target_dir (str): Path to output all of *.zip files contents.
    '''
    for f in os.listdir(source_dir):
        filepath = os.path.join(source_dir, f)
        if os.path.isfile(filepath) and fnmatch(f, '*.zip'):
            print(f"-- Extracting {f} to {target_dir}.")
            ZipFile(filepath).extractall(target_dir)


def cleanup_gtsrb_files(gtsrb_dir, target_dir):
    '''
    Moves all downloaded GTSRB files into specific directories.

    Parameters:
        gtsrb_dir (str): Path to the 'GTSRB directory'.
        target_dir (str): Path to the output directory. 
            The function will create two directories ('train', 'external') here, 
            with files downloaded from the GTSRB dataset.
    '''
    training_dir = os.path.join(target_dir, 'train')
    test_dir = os.path.join(target_dir, 'external')
    if not os.path.exists(training_dir):
        os.makedirs(training_dir, exist_ok=True)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir, exist_ok=True)
    move_files(os.path.join(
        gtsrb_dir, 'Final_Training/Images'), training_dir)
    move_files(os.path.join(gtsrb_dir, 'Final_Test/Images'), test_dir)
    os.remove(os.path.join(test_dir, 'GT-final_test.test.csv'))
    shutil.rmtree(gtsrb_dir)
    os.rename(os.path.join(target_dir, 'GT-final_test.csv'),
              os.path.join(test_dir, 'GT-final_test.csv'))


def move_files(source_dir, dest_dir, N=1, ext='*'):
    '''
    Moves specified factor of files from source_dir into dest_dir.
    Both paths must exist.

    Parameters:
        source_dir (str): Directory with *.jpg files from which files will be moved.
        dest_dir (str): Directory to which the files will be moved.
        N (int): Percentage of files to be moved. (Range: [0, 1])
        ext (str): Extension of files to be moved.
    '''
    index = 0
    size = count_files(source_dir, ext)
    for fname in os.listdir(source_dir):
        if ext == '*' or fnmatch(fname, ext):
            index += 1
            src_file = os.path.join(source_dir, fname)
            dst_file = os.path.join(dest_dir, fname)
            shutil.move(src_file, dst_file)
        if index >= int(N*size):
            break


def split_directories(source_root, dest_root, N=0.2):
    '''
    Splits *.jpg files into two directory trees. 
    Amount of files moved is specified by the N factor.
    Used for splitting the dataset into training and validation.

    Parameters:
        source_root (str): Root of the source directory tree.
        dest_root (str): Root of the destination directory tree.
        N (int): Percentage of files to be moved. (Range: [0, 1])
    '''
    for dirname in os.listdir(source_root):
        src_dir = os.path.join(source_root, dirname)
        dest_dir = os.path.join(dest_root, dirname)
        if not os.path.isdir(dest_dir):
            os.makedirs(dest_dir, exist_ok=True)
            move_files(src_dir, dest_dir, N, '*.jpg')
