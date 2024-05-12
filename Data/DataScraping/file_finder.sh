#!/bin/bash

cd ~/Downloads/CJPR/CJPR/Data/DataScraping/Yearwise_data/1951/

for file in *; do 
  grep -il "suburban" $file
done
