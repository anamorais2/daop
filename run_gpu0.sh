#!/bin/bash

echo "Launching 5 trainings in PARALLEL on GPU 0..."
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Seleciona a GPU 0 (RTX 3080)
export CUDA_VISIBLE_DEVICES=0

python main_sl_medmnist_val.py 1 > output_1.txt 2>&1 &
PID2=$!
echo "Process 1 started (PID: $PID2)"

python main_sl_medmnist_val.py 2 > output_2.txt 2>&1 &
echo "Process 2 started"

python main_sl_medmnist_val.py 3 > output_3.txt 2>&1 &
echo "Process 3 started"

python main_sl_medmnist_val.py 4 > output_4.txt 2>&1 &
echo "Process 4 started"

python main_sl_medmnist_val.py 5 > output_5.txt 2>&1 &
echo "Process 5 started"

echo "Waiting for all to finish..."
wait

echo "All 5 jobs have finished!"