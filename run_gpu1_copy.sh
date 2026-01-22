#!/bin/bash

# --- CORREÇÃO CRÍTICA PARA ESTA MÁQUINA ---
# Garante que o Python respeita a ordem do nvidia-smi
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Seleciona a GPU 1 (RTX 3080 Ti)
export CUDA_VISIBLE_DEVICES=1
# ------------------------------------------

echo "Launching 3 trainings in PARALLEL on GPU 1 (RTX 3080 Ti)..."

python main_sl_medmnist_val.py 6 > output_6_18.txt 2>&1 &
PID1=$! 
echo "Process 7 started (PID: $PID1)"

python main_sl_medmnist_val.py 7 > output_7_18.txt 2>&1 &
PID2=$!
echo "Process 8 started (PID: $PID2)"

python main_sl_medmnist_val.py 8 > output_8_18.txt 2>&1 &
echo "Process 8 started"



echo "Waiting for all to finish..."
wait

echo "All 4 jobs have finished!"