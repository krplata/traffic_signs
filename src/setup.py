import argparse
import os
import prep_for_training as prep
import sys

CLASS_SIZE = 6000


def move_files(source_dir, dest_dir, N=0.2):
    '''
    Moves specified factor of *.jpg files from source_dir into dest_dir.
    Both paths must exist.

    Parameters:
        source_dir (str): Directory with *.jpg files from which files will be moved.
        dest_dir (str): Directory to which the files will be moved.
        N (int): Percentage of files to be moved. (Range: [0, 1])
    '''
    index = 0
    for fname in os.listdir(source_dir):
        if fname.endswith('.jpg'):
            index += 1
            src_image = os.path.join(source_dir, fname)
            dst_image = os.path.join(dest_dir, fname)
            os.rename(src_image, dst_image)
        if index >= int(0.2*CLASS_SIZE):
            break


def split_directories(source_dir, dest_dir, N=0.2):
    '''
    Splits *.jpg files into two directory trees. 
    Amount of files moved is specified by the N factor.
    Used for splitting the dataset into training and validation.

    Parameters:
        source_dir (str): Root of the source directory tree.
        dest_dir (str): Root of the destination directory tree.
        N (int): Percentage of files to be moved. (Range: [0, 1])
    '''
    for root, dirnames, filenames in os.walk(source_dir):
        structure = os.path.join(dest_dir, root[len(source_dir):])
        if not os.path.isdir(structure):
            os.mkdir(structure)
            move_files(root, structure, N)


script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
parser = argparse.ArgumentParser()
parser.add_argument('--src', dest='src_path',
                    help='Sets the source path for recursive image augmentation.')
parser.add_argument('--dest', dest='dest_path',
                    help='Sets the output path for recursive image augmentation.')

args = parser.parse_args()

train_dir = os.path.join(args.dest_path, "train/")
if not os.path.exists(train_dir):
    os.makedirs(train_dir, exist_ok=True)

if not os.path.exists(args.src_path):
    print("Error: Invalid source path (prep_for_training.py)")
    exit

prep.ppm_dir_to_jpg(args.src_path, train_dir)
prep.generate_augmented(train_dir, CLASS_SIZE)
prep.cleanup_names(train_dir)

validation_dir = os.path.join(args.dest_path, "validate/")
split_directories(train_dir, validation_dir)

test_dir = os.path.join(args.dest_path, "test/")
split_directories(train_dir, test_dir)
