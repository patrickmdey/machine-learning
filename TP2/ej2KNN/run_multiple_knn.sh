#!/bin/bash

file_amount=()
test_pc=(0.1 0.2 0.3)
k_values=(5 8 10)

for i in $(seq 0 $((${#test_pc[@]}-1)))
do
    echo "{\"file\": \"reviews_sentiment.csv\",
            \"k\": 10,
            \"test_percentage\": ${test_pc[$i]},
            \"weighted\": false,
            \"normalize\": false,
            }" > config.json
    python3 main.py 
    python3 post_processing.py ${file_amount[$i]}
done