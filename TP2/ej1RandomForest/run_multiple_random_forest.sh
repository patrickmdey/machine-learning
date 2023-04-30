#!/bin/bash

tree_amount=(3 5 8 10)

for i in $(seq 0 $((${#tree_amount[@]}-1)))
do
    echo "{\"target\": \"Creditability\",
            \"file\": \"./german_credit.csv\",
            \"tree_amount\": ${tree_amount[$i]},
            \"test_percentage\": 0.2,
            \"do_forest\": true,
            \"tree_amount\": true
            }" > config.json
    python3 main.py random_forest 0.2 ${tree_amount[$i]}
    python3 post_processing.py random_forest -1 ${tree_amount[$i]}
done
