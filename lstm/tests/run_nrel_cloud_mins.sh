#!/bin/bash

# run cloud fraction prediction on HOURLY NREL data
(cd ../src/ && python driver.py\
                      -p ../data/NREL_2013_MINS_with_year.mat\
                      -f average\
                      -m cloud\
                      -o ../outputs/\
                      -n NRELMINS\
                      -s 60)
