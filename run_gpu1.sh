#!/bin/bash
echo "A iniciar GPU 1 (RTX 3080 Ti)..."
export CUDA_VISIBLE_DEVICES=1

python main_do_medmnist.py 1 &&
python main_do_medmnist.py 2 &&
python main_do_medmnist.py 3 &&
python main_do_medmnist.py 4 &&
python main_do_medmnist.py 5 &&
python main_do_medmnist.py 6

echo "Trabalho na GPU 1 terminado!"
