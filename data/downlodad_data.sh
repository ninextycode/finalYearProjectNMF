#!/usr/bin/env bash

# Indian pines
wget -nc -O 10_4231_R7RX991C.zip https://purr.purdue.edu/publications/1947/serve/1?render=archive

# MIT-CBCL
wget -nc --header="Referer: http://cbcl.mit.edu/software-datasets/heisele/download/download.html" \
    http://cbcl.mit.edu/software-datasets/heisele/download/MIT-CBCL-facerec-database.zip

# Reuters-21578
wget -nc https://archive.ics.uci.edu/ml/machine-learning-databases/reuters21578-mld/reuters21578.tar.gz

unzip 10_4231_R7RX991C.zip && mv 10_4231_R7RX991C pines
unzip MIT-CBCL-facerec-database.zip -d MIT-CBCL
mkdir -p reuters21578 && tar xkvfz reuters21578.tar.gz -C reuters21578
