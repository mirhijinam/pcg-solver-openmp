#!/bin/bash

for dir in inp/316_*; do
    if [ -d "$dir" ]; then
        dirname=$(basename "$dir")
        echo "Processing directory: $dirname"
        
        for lsf in "$dir"/*.lsf; do
            echo "submitting $lsf"
            bsub < "$lsf"
        done
        
        echo "Finished with $dirname"
        echo "-------------------"
    fi
done