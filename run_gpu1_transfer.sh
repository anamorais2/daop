#!/bin/bash

# --- CONFIG GPU ---
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# Confirma com 'nvidia-smi' se a GPU 1 é o que eu quero
export CUDA_VISIBLE_DEVICES=1

echo "Launching train in  GPU 1..."

python -u run_transfer.py > output_transfer.txt 2>&1 &

PID=$!
echo "Process initiated in background with PID: $PID"
