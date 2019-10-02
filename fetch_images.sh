#!/bin/bash

if [[ -d images ]]; then
    echo "Images directory already present"
    exit 0
elif [[ ! -d $1 ]]; then
    echo "First argument should be the project dir."
    exit 1
fi

project_dir=$1

# Download training data
wget https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip

mkdir -p $project_dir/images

unzip GTSRB_Final_Training_Images.zip
mv GTSRB/Final_Training/Images/* $project_dir/images
rm -r GTSRB
