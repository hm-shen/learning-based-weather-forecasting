#!/bin/bash

# run cloud fraction prediction on HOURLY NREL data
(cd ../src/ && python driver.py\
                      -p ../data/NREL_2013_MINS_with_year.mat\
                      -f average\
                      -m cloud\
                      -o ../outputs/\
                      -n NRELMINS\
                      --time_steps 120\
                      --ubdmin 378\
                      --lbdmax 1065\
                      -s 60)
