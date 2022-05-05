#!/bin/bash

case="result_20220502_obsbot_tstrun"

# running script for Rainfall Prediction with ConvLSTM
python ../src/main_obsbot.py --model_name seq2seq\
       --dataset radarJMA --model_mode run --data_scaling linear\
       --data_path ../data/data_kanto/ --image_size 200 --pc_size 10\
       --valid_data_path ../data/data_kanto/ \
       --train_path ../data/train_kanto_flatsampled_JMARadar.csv \
       --valid_path ../data/valid_simple_JMARadar.csv \
       --test --eval_threshold 0.5 10 20 --test_path ../data/valid_simple_JMARadar.csv \
       --result_path $case --tdim_use 12 --tdim_loss 12 --learning_rate 0.0001 --lr_decay 0.9 \
       --batch_size 2 --n_epochs 20 --n_threads 4 --checkpoint 10 \
       --loss_function MSE\
       --interp_type nearest_kdtree\
       --optimizer adam \
       --transfer_path None
