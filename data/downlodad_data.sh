#!/usr/bin/env bash

# Indian pines
wget -nc -O 10_4231_R7RX991C.zip https://purr.purdue.edu/publications/1947/serve/1?render=archive

# AT&T Laboratories Database of Faces
wget -nc http://www.cl.cam.ac.uk/Research/DTG/attarchive/pub/data/att_faces.zip

# Reuters-21578
wget -nc https://archive.ics.uci.edu/ml/machine-learning-databases/reuters21578-mld/reuters21578.tar.gz

unzip 10_4231_R7RX991C.zip -d indian_pines
(cd indian_pines && unzip bundle.zip)
(cd indian_pines/documentation &&\
    unzip NS-line_Project_and_Ground_Reference_Files.zip &&
    unzip Site3_Project_and_Ground_Reference_Files.zip)
(cd indian_pines &&\
    mkdir -p images &&\
    mv documentation/19920612_AVIRIS_IndianPine_NS-line_gr.tif images &&\
    mv documentation/Site3_Project_and_Ground_Reference_Files/19920612_AVIRIS_IndianPine_Site3_gr.tif images &&\
    mv aviris_hyperspectral_data/19920612_AVIRIS_IndianPine_NS-line.tif images
    mv aviris_hyperspectral_data/19920612_AVIRIS_IndianPine_Site3.tif images)

unzip att_faces.zip -d att_faces
(cd att_faces &&\
    mkdir -p images
    mmv s\*/\* images/\#1_\#2)

mkdir -p reuters21578 && tar xkvfz reuters21578.tar.gz -C reuters21578
