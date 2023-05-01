#!/bin/bash

max_node_amount=(50 80 100)
tree_amount=(10) #(5 10 15)
test_pc=(0.1 0.2 0.3)
partitions=(9 4 3)

for i in $(seq 0 $((${#partitions[@]}-1)))
do 
    for j in $(seq 0 $((${#max_node_amount[@]}-1)))
    do
        for k in $(seq 0 $((${#tree_amount[@]}-1)))
        do
            echo "{\"target\": \"Creditability\",
                    \"file\": \"./german_credit.csv\",
                    \"test_percentage\": ${test_pc[$i]},
                    \"max_nodes\": ${max_node_amount[$j]},
                    \"tree_amount\": ${tree_amount[$k]},
                    \"do_forest\": true
                    }" > config.json
            python3 main.py
            python3 post_processing.py random_forest ${max_node_amount[$j]} ${partitions[$i]} ${tree_amount[$k]}
        done
    done
done

for i in $(seq 0 $((${#partitions[@]}-1)))
do
    python precision_graph.py random_forest ${partitions[$i]}
done