#!/bin/bash
test_file="data/ifttt/test.bin"
model_name="model.ifttt.best.deep" # in a depth-first traversal. 
traverse_first="deep"  # breadth OR deep
python exp.py \
    --mode test \
    --evaluator ifttt_evaluator \
    --traverse_first ${traverse_first} \
    --load_model saved_models/ifttt/${model_name}.bin \
    --beam_size 15 \
    --test_file ${test_file} \
    --save_decode_to decodes/ifttt/${model_name}.test.decode \
    --decode_max_time_step 110



