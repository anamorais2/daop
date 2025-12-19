#!/bin/bash
echo "Launching 6 trainings in PARALLEL on GPU 1..."
export CUDA_VISIBLE_DEVICES=1

python main_sl_medmnist_val_thesis.py 1 > output_1.txt 2>&1 &
PID1=$! 
echo "Process 1 started (PID: $PID1)"

#python main_sl_medmnist_val_thesis.py 6 > output_6.txt 2>&1 &
#PID2=$!
#echo "Process 6 started (PID: $PID2)"

#python main_sl_medmnist_val_thesis.py 7 > output_7.txt 2>&1 &
#echo "Process 7 started"

echo "Waiting for all to finish..."
wait

echo "All 6 jobs have finished!"
