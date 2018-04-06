#!/bin/bash

# run cloud fraction prediction on HOURLY NREL data
python driver.py\
       -p ../data/NREL_total_1314_train.mat\
       -f average\
       -m cloud\
       -o ../outputs/\
       -n NREL1314\
       -s 1
