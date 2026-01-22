#!/bin/bash
echo "Launching 4 trainings in PARALLEL on GPU 1..."

# O segredo é colocar 'CUDA_VISIBLE_DEVICES=1' antes de cada comando
# Isto esconde a GPU 0 e faz a GPU 1 parecer ser a única disponível para o Python.

CUDA_VISIBLE_DEVICES=1 python main_sl_medmnist_val.py 7 > output_7.txt 2>&1 &
PID1=$! 
echo "Process 7 started (PID: $PID1) on GPU 1"

CUDA_VISIBLE_DEVICES=1 python main_sl_medmnist_val.py 8 > output_8.txt 2>&1 &
PID2=$!
echo "Process 8 started (PID: $PID2) on GPU 1"

CUDA_VISIBLE_DEVICES=1 python main_sl_medmnist_val.py 9 > output_9.txt 2>&1 &
echo "Process 9 started on GPU 1"

CUDA_VISIBLE_DEVICES=1 python main_sl_medmnist_val.py 10 > output_10.txt 2>&1 &
echo "Process 10 started on GPU 1"

echo "Waiting for all to finish..."
wait

echo "All 4 jobs have finished!"