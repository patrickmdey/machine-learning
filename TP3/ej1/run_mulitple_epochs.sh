#!/bin/bash

learning_rates=(0.1 0.01 0.001)
epochs=(100 1000 10000)
point_amount=30

python3 generate_points.py $point_amount
for i in $(seq 0 $((${#learning_rates[@]}-1)))
do
    for j in $(seq 0 $((${#epochs[@]}-1)))
    do
        echo "{
                \"generate\": false,
                \"method\": \"$1\",
                \"file_name\": \"TP3-1\",
                \"point_number\": 30,
                \"learning_rate\": ${learning_rates[$i]},
                \"epochs\": ${epochs[$j]}
                }" > config.json
        python3 main.py
        # python3 post_processing.py id3 ${max_node_amount[$j]} ${partitions[$i]}
    done
done