#!/bin/bash

THIS_DIR="$( cd "$( dirname "$0"  )" && pwd  )"
cd $THIS_DIR

CURRENT_DIR=$(pwd)




cd $CURRENT_DIR


ARCNAME=dla_34 
EXPID=coco640
EXPNAME=$ARCNAME'_'$EXPID
TASK=multi_pose_wodet
DATASET=coco_hp_wodet
RES=640
EXPDIR=$CURRENT_DIR/../exp/$TASK/$EXPNAME
mkdir -p $EXPDIR

echo "Start training"

/opt/tiger/torch12/bin/python main.py $TASK --exp_id $EXPNAME --dataset coco_hp_wodet --master_batch_size 8 --batch_size 64 --lr 2.5e-4 --gpus 0,1,2,3,4,5,6,7 \
--num_epochs 280 --lr_step 230,260 --num_workers 16  --K 20 --arch $ARCNAME \
--val_intervals 5 --input_res $RES --not_reg_hp_offset --not_reg_offset \
--aug_rot 0.5 --rotate 15 --hide_data_time

echo "Start single-scale test"

/opt/tiger/torch12/bin/python test.py $TASK --exp_id $EXPNAME --dataset $DATASET \
--resume --not_reg_offset --not_reg_hp_offset --K 20 --not_hm_hp --arch $ARCNAME --input_res $RES --keep_res

echo "Start single-scale test with flip"

/opt/tiger/torch12/bin/python test.py $TASK --exp_id $EXPNAME --dataset $DATASET \
--resume --not_reg_offset --not_reg_hp_offset --K 20 --not_hm_hp --arch $ARCNAME --input_res $RES --keep_res --flip_test 

echo "Start multi-scale test with flip"

/opt/tiger/torch12/bin/python test.py $TASK --exp_id $EXPNAME --dataset $DATASET \
--resume --not_reg_offset --not_reg_hp_offset --K 20 --not_hm_hp --arch $ARCNAME --input_res $RES --keep_res --flip_test --test_scales 1,1.25,1.5,1.75



