import argparse
import os
import sys
import requests
from zipfile import ZipFile
from tqdm import tqdm
import shutil


def fetch_file(url, target_path, filename=''):
    r = requests.get(url, stream=True)
    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024
    t = tqdm(total=total_size, unit='iB', unit_scale=True, desc=filename)
    with open(target_path, 'wb') as f:
        for data in r.iter_content(block_size):
            t.update(len(data))
            f.write(data)


def download_files(urls, dest_dir='', force=False):
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)
    for url in download_urls:
        filename = url.rsplit(sep='/', maxsplit=1)[1]
        file_path = os.path.join(dest_dir, filename)
        if not os.path.exists(file_path) or force:
            fetch_file(url, file_path, filename)
        else:
            print(f"-- File {filename} already exists. Skipping download.")


def unzip_files(source_dir, target_dir):
    for f in os.listdir(source_dir):
        filepath = os.path.join(source_dir, f)
        if os.path.isfile(filepath):
            print(f"-- Extracting {f} to {target_dir}.")
            ZipFile(filepath).extractall(target_dir)


def move_contents(source_dir, target_dir, ext=''):
    for f in os.listdir(source_dir):
        src_filepath = os.path.join(source_dir, f)
        target_filepath = os.path.join(target_dir, f)
        os.rename(src_filepath, target_filepath)


def cleanup_gtsrb_files(gtsrb_dir, target_dir):
    '''
    Moves all training and test data into specific directories.
    '''
    training_dir = os.path.join(target_dir, 'train')
    test_dir = os.path.join(target_dir, 'external')
    if not os.path.exists(training_dir):
        os.makedirs(training_dir, exist_ok=True)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir, exist_ok=True)
    move_contents(os.path.join(
        gtsrb_dir, 'Final_Training/Images'), training_dir)
    move_contents(os.path.join(gtsrb_dir, 'Final_Test/Images'), test_dir)
    os.remove(os.path.join(test_dir, 'GT-final_test.test.csv'))


parser = argparse.ArgumentParser()
parser.add_argument('--skip-external', dest='skip_ext',
                    help='Skips downloading of external test data for the recognition stage.')
parser.add_argument('--skip-alignment', dest='skip_align',
                    help='Skips the alignment of classes in the recognition stage training data.')
args = parser.parse_args()


test_url1 = 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip'
test_url2 = 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip'
recognition_train_url = 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip'
detection_full_url = 'https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/FullIJCNN2013.zip'

download_urls = [recognition_train_url]
if not args.skip_ext:
    download_urls.extend([test_url1, test_url2])

project_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
archive_dir = os.path.join(project_dir, 'archives')
data_dir = os.path.join(project_dir, 'data')
gtsrb_dir = os.path.join(data_dir, 'GTSRB')

download_files(download_urls, archive_dir)
unzip_files(archive_dir, data_dir)

cleanup_gtsrb_files(gtsrb_dir, data_dir)
shutil.rmtree(gtsrb_dir)
os.rename(os.path.join(data_dir, 'GT-final_test.csv'),
          os.path.join(data_dir, 'external', 'GT-final_test.csv'))
