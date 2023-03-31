#!/bin/bash

THIS_DIR="$( cd "$( dirname "$0"  )" && pwd  )"
# cd $THIS_DIR
# cd /opt/tiger/
# hdfs dfs -get hdfs://haruna/home/byte_arnold_hl_vc/xiaoyabo/torch12.tar.gz
# tar zxvf torch12.tar.gz

cd $THIS_DIR
CURRENT_DIR=$(pwd)

# cd $CURRENT_DIR
# source prepare_env.sh
# cd $CURRENT_DIR/../cocoapi/PythonAPI
# make
# /opt/tiger/torch12/bin/python setup.py install --user


# cd $CURRENT_DIR/../CrowdPose/crowdpose-api/PythonAPI
# make install
# /opt/tiger/torch12/bin/python setup.py install --user


# cd $CURRENT_DIR/lib/models/networks/DCNv2
# /opt/tiger/torch12/bin/python setup.py build develop
# cd $CURRENT_DIR/lib/external
# make
# cd $CURRENT_DIR/lib/models/resample2d_package
# /opt/tiger/torch12/bin/python setup.py install --user

# cd $CURRENT_DIR
# source prepare_data.sh

cd $CURRENT_DIR


ARCNAME=hrnet_48
EXPID=crowdpose640
EXPNAME=$ARCNAME'_'$EXPID
TASK=multi_pose_crowdpose
DATASET=crowdpose
RES=640
EXPDIR=$CURRENT_DIR/../exp/$TASK/$EXPNAME
mkdir -p $EXPDIR

echo "Start training"

/opt/tiger/torch12/bin/python main.py $TASK --exp_id $EXPNAME --dataset $DATASET --master_batch_size 8 --batch_size 64 --lr 2.5e-4 --gpus 0,1,2,3,4,5,6,7 \
--num_epochs 280 --lr_step 230,260 --num_workers 16  --K 20 --arch $ARCNAME \
--val_intervals 5 --input_res $RES --not_reg_hp_offset --not_reg_offset \
--aug_rot 0.5 --rotate 15 --hide_data_time

echo "Start single-scale test"

/opt/tiger/torch12/bin/python test.py $TASK --exp_id $EXPNAME --dataset $DATASET \
--resume --not_reg_offset --not_reg_hp_offset --K 20 --not_hm_hp --arch $ARCNAME --input_res $RES --keep_res --trainval

echo "Start single-scale test with flip"

/opt/tiger/torch12/bin/python test.py $TASK --exp_id $EXPNAME --dataset $DATASET \
--resume --not_reg_offset --not_reg_hp_offset --K 20 --not_hm_hp --arch $ARCNAME --input_res $RES --keep_res --flip_test --trainval

echo "Start multi-scale test with flip"

/opt/tiger/torch12/bin/python test.py $TASK --exp_id $EXPNAME --dataset $DATASET \
--resume --not_reg_offset --not_reg_hp_offset --K 20 --not_hm_hp --arch $ARCNAME --input_res $RES --keep_res --trainval --flip_test --test_scales 0.8,1,1.2,1.4,1.6



