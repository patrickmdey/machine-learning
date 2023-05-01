#!/bin/bash

max_node_amount=(-1 10 30 50 80 100)
test_pc=(0.1 0.2 0.3)
partitions=(9 4 3)

for i in $(seq 0 $((${#partitions[@]}-1)))
do 
    for j in $(seq 0 $((${#max_node_amount[@]}-1)))
    do
        echo "{\"target\": \"Creditability\",
                \"file\": \"./german_credit.csv\",
                \"max_nodes\": ${max_node_amount[$j]},
                \"test_percentage\": ${test_pc[$i]},
                \"do_forest\": false
                }" > config.json
        python3 main.py
        python3 post_processing.py id3 ${max_node_amount[$i]} ${partitions[$i]}
    done
done