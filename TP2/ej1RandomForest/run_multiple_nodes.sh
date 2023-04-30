#!/bin/bash

max_node_amount=(-1 10 30 50 80 100)
test_pc=(0.1 0.2 0.3)

for i in $(seq 0 $((${#max_node_amount[@]}-1)))
do
    echo "{\"target\": \"Creditability\",
            \"file\": \"./german_credit.csv\",
            \"max_nodes\": ${max_node_amount[$i]},
            \"test_percentage\": 0.2,
            \"do_forest\": false
            }" > config.json
    python3 main.py #id3 0.2 ${max_node_amount[$i]}
    python3 post_processing.py id3 ${max_node_amount[$i]} 9
done