#!/bin/bash

kernels=("linear" "poly" "rbf" "sigmoid")
c_values=(0.1 1.6 2)

for i in $(seq 0 $((${#c_values[@]} - 1))); do
    for j in $(seq 0 $((${#kernels[@]} - 1))); do
        echo "{
                \"image_file\": \"cow\",
                \"c_value\": ${c_values[$i]},
                \"kernel\": \"${kernels[$j]}\"
                }" >classify_config.json
        python3 classify.py
    done
done
