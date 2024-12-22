# Conditional Consistency Models

Welcome to the official repository for Conditional Consistency Models (CCM). This repository hosts the implementation and evaluation of consistency models tailored for various datasets, including **BCI**, **LLVIP**, **LOLv1**, **LOLv2**, and **SID**.

This repository contains code for training and evaluating models, as well as pre-trained model weights and metrics calculation scripts.

## Table of Contents

1. [Requirements and Setup](#requirements-and-setup)
2. [Directory Structure](#directory-structure)
3. [Datasets](#datasets)
4. [Pre-trained Model Checkpoints](#pre-trained-model-checkpoints)
5. [Usage](#usage)
6. [Citation](#citation)

## Requirements and Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Conditional-Consistency-Models.git
cd Conditional-Consistency-Models
```

### 2. Create a Conda Environment
Create and activate a new conda environment named ccm:
```bash
conda create -n ccm python=3.10 -y
conda activate ccm
```

### 3. Install Dependencies
Install all required Python packages:
```bash
pip install -r requirements.txt
```

### 4. Install Additional Tools
Make sure you have git-lfs installed to handle large files (if applicable):
```bash
conda install -c conda-forge git-lfs
git lfs install
```

## Directory Structure

Ensure your project directory is organized as follows:

```
Conditional-Consistency-Models/
    ├── datasets/                 # Folder for datasets
    │   ├── bci/
    │   ├── llvip/
    │   ├── lolv1/
    │   ├── lolv2/
    │   ├── sid/
    ├── bci/                      # Folder for BCI model scripts and metrics
    │   ├── script.py
    │   ├── sampling_and_metric.py
    │   ├── checkpoints/          # Pre-trained BCI model weights
    │       ├── model.json
    │       ├── model.pt
    ├── llvip/                    # Folder for LLVIP scripts
    │   ├── script.py
    │   ├── metrics.py
    │   ├── checkpoints/
    ├── lolv1/
    ├── lolv2/
    ├── sid/
    ├── improved_consistency_model_conditional.py
    ├── README.md
    ├── requirements.txt
```

## Datasets

### 1. Download Datasets

Use the links below to download datasets for each model. Once downloaded, extract and place them inside the `datasets/` directory.

- BCI Dataset: [Insert BCI Dataset Link]
- LLVIP Dataset: [Insert LLVIP Dataset Link]
- LOLv1 Dataset: [Insert LOLv1 Dataset Link]
- LOLv2 Dataset: [Insert LOLv2 Dataset Link]
- SID Dataset: [Insert SID Dataset Link]

### 2. Dataset Organization

Organize each dataset in the following format:

#### BCI Dataset
```
datasets/bci/
    ├── HE/
        ├── train/
        ├── test/
    ├── IHC/
        ├── train/
        ├── test/
```

#### LLVIP Dataset
```
datasets/llvip/
    ├── visible/
        ├── train/
        ├── test/
    ├── infrared/
        ├── train/
        ├── test/
```

#### LOLv1 Dataset
```
datasets/lolv1/
    ├── eval_new/
        ├── low/
        ├── high/
```

#### LOLv2 Dataset
```
datasets/lolv2/
    ├── train/
    ├── test/
```

#### SID Dataset
```
datasets/sid/
    ├── train/
    ├── test/
```

## Pre-trained Model Checkpoints

Download the pre-trained model weights for each dataset from the links below and place them in the corresponding `checkpoints/` folder:

- BCI Model Checkpoints: [Insert Link]
- LLVIP Model Checkpoints: [Insert Link]
- LOLv1 Model Checkpoints: [Insert Link]
- LOLv2 Model Checkpoints: [Insert Link]
- SID Model Checkpoints: [Insert Link]

Example:
```
bci/checkpoints/
    ├── model.json
    ├── model.pt
```

## Usage

### 1. Training

To train a model on a specific dataset, use the corresponding `script.py` inside the dataset's folder.

For example, to train the BCI model:
```bash
cd bci
CUDA_VISIBLE_DEVICES=0 python -m bci.script
```

### 2. Evaluation (Metrics)

To evaluate the model and calculate PSNR/SSIM metrics, use the `sampling_and_metric.py` script.

Example: Evaluate the BCI model:
```bash
cd bci
CUDA_VISIBLE_DEVICES=0 python -m bci.sampling_and_metric
```

Ensure the dataset is correctly placed, and pre-trained checkpoints are available before running the evaluation.

### Results

Evaluation metrics (PSNR, SSIM) and generated images are saved in the corresponding output folder, e.g., `lolv1_final/`.

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{yourpaper2024,
    author = {Your Name and Collaborators},
    title = {Conditional Consistency Models},
    journal = {Conference Name},
    year = {2024},
    url = {https://github.com/your-username/Conditional-Consistency-Models},
}
```

For any issues, feel free to open an issue on this repository.
