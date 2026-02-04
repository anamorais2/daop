# DAOP — Data Augmentation Optimizer (Supervised Learning)

This repository contains an implementation of the **Data Augmentation Optimizer (DAOP)** framework applied to **Supervised Learning** on the **MedMNIST v2** benchmark.

DAOP is an evolutionary framework that optimizes data augmentation policies online during model training. While the original framework was introduced for Self-Supervised Learning (DAOP4SSL), this repository provides its adaptation to a fully supervised medical imaging setting.

---

## Framework Overview

DAOP is composed of three main modules:

- **ES Module** — evolves augmentation policies using an evolutionary strategy
- **DA Module** — applies candidate augmentation policies online during training
- **ML Module** — trains the classifier and returns validation performance as fitness

This repository implements this pipeline for:
- MedMNIST datasets (BreastMNIST, DermaMNIST, PneumoniaMNIST, OrganCMNIST)
- ResNet-18 and ResNet-50 architectures
- Low-resolution optimization (28×28) and transfer experiments to 224×224

---

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)

### Step 1 — Create a Virtual Environment (recommended)

```bash
# Using conda
conda create -n daop_sl python=3.8
conda activate daop_sl

# Or using venv
python -m venv daop_env
source daop_env/bin/activate  # Windows: daop_env\Scripts\activate
```

### Step 2 — Install PyTorch

Install PyTorch before other dependencies to ensure correct CUDA compatibility.

Visit: https://pytorch.org/get-started/locally/

Or use one of the following:

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch torchvision
```

### Step 3 — Install Remaining Dependencies

```bash
pip install -r requirements.txt
```

---

## Running Experiments

The main entry point for supervised experiments is:

```bash
python main_sl_medmnist_val.py
```

or with a number appended, which represents a specific seed:

```bash
python main_sl_medmnist_val.py 1 
```

### Example — BreastMNIST with default settings

```bash
python main_sl_medmnist_val.py --dataset breastmnist
```


## Configuration

The framework is highly configurable. Most experiment settings can be modified through command-line arguments or configuration files in `configs/`, including:

- Dataset selection
- Model architecture (ResNet-18, ResNet-50)
- Evolutionary parameters (generations, population size, mutation rates)
- Training hyperparameters (learning rate, batch size, epochs)

This allows easy adaptation to new datasets, models, or augmentation operators.

---

## Relation to the Original DAOP Framework (SSL)

This repository is based on the original DAOP framework designed for Self-Supervised Learning (DAOP4SSL). If you are interested in the SSL version, refer to the original repository where DAOP is used for:

- **Pretext Optimization (PO)** — optimizing augmentations for pretext tasks
- **Downstream Optimization (DO)** — optimizing augmentations for fine-tuning
- **Simultaneous PO + DO** — joint optimization

The present repository focuses **exclusively** on the Supervised Learning instantiation for medical imaging.

## Troubleshooting

**CUDA Out of Memory:**

 - Reduce batch_size
 
```bash
python main_sl_medmnist_val.py --dataset breastmnist --batch_size 32
```

**Check CUDA availability:**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**Verify GPU usage:**
```bash
nvidia-smi
```
