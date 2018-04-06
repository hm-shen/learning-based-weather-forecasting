#!/bin/bash

# use LSTM model to forecast solar irradiance using WRF dataset
(cd ../src/ && python driver.py\
                      -p ../data/WRF_data_2017_Jan_to_Mar.mat\
                      -f average\
                      -m solar\
                      -o ../outputs/\
                      -n WRF_Jan_to_Mar\
                      -s 1)
