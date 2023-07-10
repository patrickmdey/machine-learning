#!/bin/bash

test_pc=(0.1 0.2 0.3)
partitions=(9 4 3)

n_estimators=(5 10 15)

echo "Simulating all configurations..."
for i in $(seq 0 $((${#partitions[@]} - 1))); do
    for j in $(seq 0 $((${#n_estimators[@]} - 1))); do
        echo "{\"target\": \"HDisease\",
                \"file\": \"./heart.csv\",
                \"partitions\": ${partitions[$i]},
                \"test_size\": ${test_pc[$i]},
                \"n_estimators\": ${n_estimators[$j]}
                }" >config.json
        python3 main.py
    done
done

echo "Plotting score graphs..."

for i in $(seq 0 $((${#partitions[@]} - 1))); do
    python3 plot_scores.py ${partitions[$i]}
done
