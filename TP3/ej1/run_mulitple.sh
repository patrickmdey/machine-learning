#!/bin/bash

learning_rates=(0.1 0.01 0.001)
epochs=(100 1000 10000)
point_amounts=(30 60 120)

for k in $(seq 0 $((${#point_amounts[@]}-1)))
do
    python3 generate_points.py ${point_amounts[$k]}    
    for i in $(seq 0 $((${#learning_rates[@]}-1)))
    do
        for j in $(seq 0 $((${#epochs[@]}-1)))
        do
            echo "{
                    \"generate\": false,
                    \"method\": \"$1\",
                    \"file_name\": \"TP3-1\",
                    \"point_number\": ${point_amounts[$k]},
                    \"learning_rate\": ${learning_rates[$i]},
                    \"epochs\": ${epochs[$j]}
                    }" > config.json
            python3 main.py
        done
    done
done