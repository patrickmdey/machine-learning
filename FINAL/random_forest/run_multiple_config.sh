#!/bin/bash

test_pc=(0.1 0.2 0.3)
partitions=(10 5 3)

tree_amount=(5 10 15)

echo "Simulating all configurations..."
for i in $(seq 0 $((${#partitions[@]} - 1))); do
    for j in $(seq 0 $((${#tree_amount[@]} - 1))); do
        echo "{\"target\": \"HDisease\",
                \"file\": \"./heart.csv\",
                \"partitions\": ${partitions[$i]},
                \"test_percentage\": ${test_pc[$i]},
                \"tree_amount\": ${tree_amount[$j]}
                }" >config.json
        python3 main.py
    done
done

echo "Plotting score graphs..."

for i in $(seq 0 $((${#partitions[@]} - 1))); do
    python3 plot_scores.py ${partitions[$i]}
done

echo "Best configuration..."
python3 print_best_config.py
