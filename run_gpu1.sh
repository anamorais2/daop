#!/bin/bash

echo "Launching 5 trainings in PARALLEL on GPU 1.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Seleciona a GPU 0 (RTX 3080)
export CUDA_VISIBLE_DEVICES=1

python main_sl_medmnist_val_copy.py 6 > output_6.txt 2>&1 &
PID2=$!
echo "Process 1 started (PID: $PID2)"

python main_sl_medmnist_val_copy.py 7 > output_7.txt 2>&1 &
echo "Process 2 started"

python main_sl_medmnist_val_copy.py 8 > output_8.txt 2>&1 &
echo "Process 3 started"

python main_sl_medmnist_val_copy.py 9 > output_9.txt 2>&1 &
echo "Process 4 started"

python main_sl_medmnist_val_copy.py 10 > output_10.txt 2>&1 &
echo "Process 5 started"

echo "Waiting for all to finish..."
wait

echo "All 5 jobs have finished!"