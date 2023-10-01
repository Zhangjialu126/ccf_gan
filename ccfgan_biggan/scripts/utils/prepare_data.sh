#!/bin/bash
#python make_hdf5.py --dataset I128 --batch_size 256 --data_root ./data
#CUDA_VISIBLE_DEVICES=0 python calculate_inception_moments.py --dataset C10 --data_root ./data
#CUDA_VISIBLE_DEVICES=0 python calculate_inception_moments.py --dataset V200 --data_root ./data
#CUDA_VISIBLE_DEVICES=0 python calculate_inception_moments.py --dataset V500 --data_root ./data
#CUDA_VISIBLE_DEVICES=0 python calculate_inception_moments.py --dataset V1000 --data_root ./data
CUDA_VISIBLE_DEVICES=0 python calculate_inception_moments.py --dataset I128 --data_root ./data