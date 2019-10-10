#!/bin/bash

test_url=''
training_url='https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip'
fetch_url="$training_url"
project_dir="$(realpath $(dirname $0))"

if [[ -t 1 ]]; then
    ncolors=$(tput colors)
    if test -n "$ncolors" && test $ncolors -ge 8; then
        bold="$(tput bold)"
        normal="$(tput sgr0)"
        black="$(tput setaf 0)"
        red="$(tput setaf 1)"
        green="$(tput setaf 2)"
        yellow="$(tput setaf 3)"
        blue="$(tput setaf 4)"
        magenta="$(tput setaf 5)"
        cyan="$(tput setaf 6)"
        white="$(tput setaf 7)"
    fi
fi

p_error() {
    echo -e "${red}---> [Error]${white} $1"
}

p_info_msg() {
    echo -e "${green}--->${white} $1"
}

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
if [[ -f "$project_dir/$filename" ]]; then
    p_info_msg "File already exists, skipping download"
else
    p_info_msg "Downloading $filename"
    wget $fetch_url
fi

unzip -qu "$project_dir/$filename"
python -m venv traffic_signs
source traffic_signs/bin/activate
pip install -r requirements.txt

p_info_msg "Setting up the dataset at: $project_dir/data"
python $project_dir/src/setup.py --src "$project_dir/GTSRB/Final_Training/Images" \
    --dest "$project_dir/data"
