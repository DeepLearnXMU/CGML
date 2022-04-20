#!/bin/bash
test_file="data/atis/test.bin"
model_name="model.atis.best.deep" # in a breadth-first manner. 
traverse_first="deep" # breadth OR deep
python exp.py \
    --mode test \
    --traverse_first ${traverse_first} \
    --load_model saved_models/atis/${model_name}.bin \
    --beam_size 5 \
    --test_file ${test_file} \
    --save_decode_to decodes/atis/${model_name}.test.decode \
    --decode_max_time_step 110



