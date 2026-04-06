#!/bin/bash

# Loop through directories A_0 to A_7
for i in {0..7}
do
    mv A_$i/boozmn_temp.nc A_$i/boozmn_out.nc
done
