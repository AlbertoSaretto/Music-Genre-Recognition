#!/bin/bash

# Script to run the python script for adding training_{i}.npy files to the h5 file containing the datset
# Run it as: bash bash_script_create_h5_train_auto.sh (or bash nameofthisfile.sh)


for j in {2..16}
do
    echo $i
    python class_CreateTrain.py $j # possibly this is called create_h5_from_npy.py now
done
