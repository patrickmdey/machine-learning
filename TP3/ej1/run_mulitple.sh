#!/bin/bash

learning_rates=(0.1 0.01 0.001)
epochs=(1000 10000 100000)
point_amounts=(30 60 120)

if (($# != 2)); then
    echo "Usage: ./run_multiple.sh <method> <error_rate>"
    exit 1
fi

error_rate=$2
file_name="input/no_noise/TP3-1"

if [ $(echo "$error_rate == 0.05" | bc -l) -eq 1 ]; then
    file_name="input/with_noise/TP3-2"
fi

for k in $(seq 0 $((${#point_amounts[@]} - 1))); do
    # if [ $1 == "perceptron" ]; then
    #     python3 generate_points.py $file_name ${point_amounts[$k]} $error_rate
    # fi
    for i in $(seq 0 $((${#learning_rates[@]} - 1))); do
        for j in $(seq 0 $((${#epochs[@]} - 1))); do
            echo "{
                    \"generate\": false,
                    \"method\": \"$1\",
                    \"file_name\": \"$file_name\",
                    \"point_amount\": ${point_amounts[$k]},
                    \"learning_rate\": ${learning_rates[$i]},
                    \"epochs\": ${epochs[$j]},
                    \"error_rate\": $error_rate
                    }" >config.json
            python3 main.py
        done
    done
done
