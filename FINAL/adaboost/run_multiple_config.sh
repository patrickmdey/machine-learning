#!/bin/bash

test_pc=(0.1 0.2 0.3)
partitions=(10 5 3)

n_estimators=(50 100 150)
learning_rate=(0.1 0.5 0.7)

echo "Simulating all configurations..."
for i in $(seq 0 $((${#partitions[@]} - 1))); do
    for j in $(seq 0 $((${#n_estimators[@]} - 1))); do
        for k in $(seq 0 $((${#learning_rate[@]} - 1))); do
            echo "{\"target\": \"HDisease\",
                    \"file\": \"./heart.csv\",
                    \"partitions\": ${partitions[$i]},
                    \"test_size\": ${test_pc[$i]},
                    \"n_estimators\": ${n_estimators[$j]},
                    \"learning_rate\": ${learning_rate[$k]}
                    }" >config.json
            python3 main.py
        done
    done
done

echo "Plotting score graphs..."

for i in $(seq 0 $((${#partitions[@]} - 1))); do
    for j in $(seq 0 $((${#learning_rate[@]} - 1))); do
        python3 plot_scores.py ${partitions[$i]} ${learning_rate[$j]}
    done
done

echo "Best configuration..."
python3 print_best_config.py
