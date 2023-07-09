#!/bin/bash

test_pc=(0.1 0.2 0.3)
partitions=(9 4 3)

n_estimators=(20 50 70)
learning_rate=(0.5 1.0 1.5)

# for i in $(seq 0 $((${#partitions[@]} - 1))); do
#     for j in $(seq 0 $((${#n_estimators[@]} - 1))); do
#         for k in $(seq 0 $((${#learning_rate[@]} - 1))); do
#             echo "{\"target\": \"Creditability\",
#                     \"file\": \"./german_credit.csv\",
#                     \"test_percentage\": ${test_pc[$i]},
#                     \"n_estimators\": ${n_estimators[$j]},
#                     \"learning_rate\": ${learning_rate[$k]}
#                     }" >config.json
#             python3 main.py
#             python3 post_processing.py ${partitions[$i]} ${n_estimators[$j]} ${learning_rate[$k]}
#         done
#     done
# done

for i in $(seq 0 $((${#partitions[@]} - 1))); do
    for j in $(seq 0 $((${#learning_rate[@]} - 1))); do
        python3 precision_graph.py ${partitions[$i]} ${learning_rate[$j]}
    done
done
