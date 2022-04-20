#!/bin/bash
test_file="data/geo/test.bin"
model_name="model.geo.best.deep" # in a depth-first traversal. 
traverse_first="deep" # breadth OR deep

python exp.py \
    --mode test \
    --traverse_first ${traverse_first} \
    --load_model saved_models/geo/${model_name}.bin \
    --beam_size 5 \
    --test_file ${test_file} \
    --save_decode_to decodes/geo/${model_name}.test.decode \
    --decode_max_time_step 110


