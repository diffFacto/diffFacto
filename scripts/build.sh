#!bin/bash 

cd ./anchor_diff/metrics/emd 
python setup.py install 
cd ../chamfer_dist
python setup.py install
cd ../../../
