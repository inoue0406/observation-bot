#!/bin/bash

case="result_20220505_obsbot_artfield_fixed"

# running script for Rainfall Prediction with ConvLSTM
python ../src/main_obsbot.py --model_name seq2seq\
       --dataset artfield --model_mode run --data_scaling linear\
       --data_path ../data/artfield/vzero/ --image_size 200 --pc_size 10\
       --valid_data_path ../data/artfield/vzero/ \
       --train_path ../data/artificial_uniform_train.csv \
       --valid_path ../data/artificial_uniform_valid.csv \
       --test --eval_threshold 0.5 10 20 --test_path ../data/artificial_uniform_valid.csv \
       --result_path $case --tdim_use 12 --tdim_loss 12 --learning_rate 0.0001 --lr_decay 0.9 \
       --batch_size 2 --n_epochs 20 --n_threads 4 --checkpoint 10 \
       --loss_function MSE\
       --interp_type nearest_kdtree\
       --optimizer adam \
       --transfer_path None
