#!/bin/bash

tree_amount=(5 10 15)
test_pc=(0.1 0.2 0.3)
partitions=(9 4 3)

for i in $(seq 0 $((${#partitions[@]}-1)))
do
    for j in $(seq 0 $((${#tree_amount[@]}-1)))
    do
        echo "{\"target\": \"Creditability\",
                \"file\": \"./german_credit.csv\",
                \"tree_amount\": ${tree_amount[$j]},
                \"test_percentage\": ${test_pc[$i]},
                \"do_forest\": true
                }" > config.json
        python3 main.py random_forest
        python3 post_processing.py random_forest -1 ${partitions[$i]} ${tree_amount[$j]}
    done
done
