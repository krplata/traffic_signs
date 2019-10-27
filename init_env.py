import argparse
import os
import sys
from src import dataset_utils as utils
import shutil
from src import augmentation as aug

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
reco_data_dir = os.path.join(project_dir, 'data', 'recognition')
gtsrb_dir = os.path.join(reco_data_dir, 'GTSRB')
train_dir = os.path.join(reco_data_dir, 'train')

utils.download_files(download_urls, archive_dir)
utils.unzip_files(archive_dir, reco_data_dir)

utils.cleanup_gtsrb_files(gtsrb_dir, reco_data_dir)

aug.ppm_dir_to_grey_jpg(reco_data_dir, train_dir)

recognition_class_size = 4000
aug.generate_augmented(train_dir, recognition_class_size)
utils.cleanup_augmentor_names(train_dir)

utils.split_directories(train_dir, os.path.join(reco_data_dir, "validate/"))
utils.split_directories(train_dir, os.path.join(reco_data_dir, "test/"))
