#!/bin/bash


dataDirName="./dataset/" # Dataset directory path
source_X=$dataDirName"source_X.csv" # source dataset having feature values
source_y=$dataDirName"source_Y.csv" # source dataset having subtype labels in integer
target_X=$dataDirName"target_X.csv" # target dataset having domain index
target_test=$dataDirName"target_test.csv" # test dataset included in the target dataset

python3 bctypeFinder.py $source_X $source_y $target_X $target_test