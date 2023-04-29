#!/bin/bash

tree_amount=(3 5 8 10)

for i in $(seq 0 $((${#tree_amount[@]}-1)))
do
    echo "* Running with tree_amount = ${tree_amount[$i]}"
    python3 main.py random_forest ${tree_amount[$i]}
    python3 post_processing.py random_forest -1 ${tree_amount[$i]}
done
