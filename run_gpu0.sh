#!/bin/bash
echo "A iniciar GPU 0 (RTX 3080)..."
export CUDA_VISIBLE_DEVICES=0

python main_do_medmnist.py 7 &&
python main_do_medmnist.py 8 &&
python main_do_medmnist.py 9 &&
python main_do_medmnist.py 10

echo "Trabalho na GPU 0 terminado!"
