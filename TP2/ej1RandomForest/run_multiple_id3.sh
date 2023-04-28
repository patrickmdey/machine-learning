#!/bin/bash

max_node_amount=(-1 10 30 50 80 100)

for i in $(seq 0 $((${#max_node_amount[@]}-1)))
do
    echo "Running with max_node_amount = ${max_node_amount[$i]}"
    python3 main.py id3 ${max_node_amount[$i]}
done
