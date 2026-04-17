#!/bin/bash

echo "Launching 5 trainings in PARALLEL on GPU 0..."
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Seleciona a GPU 0 (RTX 3080)
export CUDA_VISIBLE_DEVICES=0

python main_sl_medmnist_val.py 6 > output_derma_6.txt 2>&1 &
PID2=$!
echo "Process 1 started (PID: $PID2)"

python main_sl_medmnist_val.py 7 > output_derma_7.txt 2>&1 &
echo "Process 2 started"

python main_sl_medmnist_val.py 8 > output_derma_8.txt 2>&1 &
echo "Process 3 started"

python main_sl_medmnist_val.py 9 > output_derma_9.txt 2>&1 &
echo "Process 4 started"

python main_sl_medmnist_val.py 10 > output_derma_10.txt 2>&1 &
echo "Process 5 started"

#echo "Waiting for all to finish..."
#wait

echo "All 5 jobs have finished!"