#!/bin/bash

echo "Launching 4 trainings in PARALLEL on GPU 0..."
export CUDA_VISIBLE_DEVICES=0

python main_do_medmnist_val.py 1 > output_1.txt 2>&1 &
PID1=$! 
echo "Process 1 started (PID: $PID1)"

python main_do_medmnist_val.py 2 > output_2.txt 2>&1 &
PID2=$!
echo "Process 2 started (PID: $PID2)"

python main_do_medmnist_val.py 3 > output_3.txt 2>&1 &
echo "Process 3 started"

python main_do_medmnist_val.py 4 > output_4.txt 2>&1 &
echo "Process 4 started"

echo "Waiting for all to finish..."
wait

echo "All 4 jobs have finished!"