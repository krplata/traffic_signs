#!/bin/bash

usage(){

}

test_url=''
training_url='https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip'
fetch_url="$training_url"
project_dir=$(realpath .)

while getopts ":tc:d:" opt; do
    case ${opt} in
    t) fetch_url="$test_url" ;;
    c) fetch_url="$OPTARG" ;;
    d) project_dir="$OPTARG"
    *)
        p_error "[Error] Invalid argument $OPTARG"
        usage
        exit 1
        ;;
    esac
done

# Check if path $project_dir/images exists, is a directory and isn't empty

filename="$(echo $fetch_url | grep -Eo '[a-zA-Z0-9._]+$')"
wget $fetch_url
mkdir -p $project_dir/images
mv $filename $project_dir/images
unzip -q "$project_dir/images/$filename"