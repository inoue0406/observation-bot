#!/bin/bash

case="result_20221231_obsbot_nopolicy"

# Running Obsbot Script
python ../src/main_obsbot.py --model_name obsbot\
       --dataset artfield --model_mode run --data_scaling linear\
       --image_size 256 \
       --pc_size 10 --pc_initialize random\
       --train_path ../data/artificial_uniform_train.csv \
       --data_path ../data/artfield/vzero_256/ \
       --valid_path ../data/artificial_uniform_valid.csv \
       --valid_data_path ../data/artfield/vzero/ \
       --test_path ../data/artificial_uniform_valid.csv \
       --result_path $case --tdim_use 12 --tdim_loss 12 \
       --learning_rate 0.01 --lr_decay 0.9 \
       --batch_size 8 --n_epochs 300 --n_threads 4 --checkpoint 100 \
       --loss_function MSE \
       --observer_type conv2d --policy_type nopolicy --predictor_type deconv2d \
       --freeze 1 0 0 \
       --observer_transfer_path ../run/result_20221230_observer_only/trained_observer.dict \
       --interp_type nearest_kdtree \
       --optimizer adam

# post plotting
python ../post/plot_pred_artfield_obsbot.py $case
# gif animation
python ../post/gif_animation.py $case

