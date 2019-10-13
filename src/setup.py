import argparse
import os
import prep_for_training as prep
import sys

CLASS_SIZE = 4000


def move_files(source_dir, dest_dir):
    index = 0
    for fname in os.listdir(source_dir):
        if fname.endswith('.jpg'):
            index += 1
            src_image = os.path.join(source_dir, fname)
            dst_image = os.path.join(dest_dir, fname)
            os.rename(src_image, dst_image)
        if index >= int(0.2*CLASS_SIZE):
            break


def recreate_dir_tree(source_dir, dest_dir):
    for root, dirnames, filenames in os.walk(source_dir):
        structure = os.path.join(dest_dir, root[len(source_dir):])
        if not os.path.isdir(structure):
            os.mkdir(structure)
            move_files(root, structure)


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
recreate_dir_tree(train_dir, validation_dir)
