#!/bin/bash

test_url=''
training_url='https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip'
fetch_url="$training_url"
project_dir=$(realpath .)

while getopts ":tc:d:" opt; do
    case ${opt} in
    t) fetch_url="$test_url" ;;
    c) fetch_url="$OPTARG" ;;
    d) project_dir="$OPTARG" ;;
    *)
        p_error "[Error] Invalid argument $OPTARG"
        usage
        exit 1
        ;;
    esac
done

filename="$(echo $fetch_url | grep -Eo '[a-zA-Z0-9._]+$')"
#wget $fetch_url
#unzip -q "$project_dir/$filename"
python -m venv traffic_signs
source traffic_signs/bin/activate
pip install -r requirements.txt
python $project_dir/src/prep_for_training.py
