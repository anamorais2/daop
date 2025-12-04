#!/bin/bash
echo "Launching 6 trainings in PARALLEL on GPU 1..."
export CUDA_VISIBLE_DEVICES=1

python main_sl_medmnist_val.py 5 > output_5.txt 2>&1 &
PID1=$! 
echo "Process 5 started (PID: $PID1)"

python main_sl_medmnist_val.py 6 > output_6.txt 2>&1 &
PID2=$!
echo "Process 6 started (PID: $PID2)"

python main_sl_medmnist_val.py 7 > output_7.txt 2>&1 &
echo "Process 7 started"

python main_sl_medmnist_val.py 8 > output_8.txt 2>&1 &
echo "Process 8 started"

python main_sl_medmnist_val.py 9 > output_9.txt 2>&1 &
echo "Process 9 started"

python main_sl_medmnist_val.py 10 > output_10.txt 2>&1 &
echo "Process 10 started"

echo "Waiting for all to finish..."
wait

echo "All 6 jobs have finished!"
