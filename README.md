# Conditional Consistency Models

Welcome to the official repository for Conditional Consistency Models (CCM). This repository hosts the implementation and evaluation of consistency models tailored for various datasets, including **IRVI**, **BCI**, **LLVIP**, **LOLv1**, **LOLv2**, and **SID**.

This repository contains code for training and evaluating models.

## Table of Contents

1. [Requirements and Setup](#requirements-and-setup)
2. [Directory Structure](#directory-structure)
3. [Datasets](#datasets)
4. [Usage](#usage)


## Requirements and Setup

### 1. Clone the Repository
```bash
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
    │   ├── irvi/
    ├── bci/                      # Folder for BCI model scripts and metrics
    │   ├── script.py
    │   ├── sampling_and_metric.py
    │   ├── checkpoints/bci/          # Pre-trained BCI model weights
    │       ├── model.json
    │       ├── model.pt
    ├── llvip/                   
    ├── lolv1/
    ├── lolv2/
    ├── sid/
    ├── irvi/
    ├── improved_consistency_model_conditional.py
    ├── README.md
    ├── requirements.txt
```

## Datasets

### Download Datasets

Use the links below to download datasets for each model. Once downloaded, extract and place them inside the `datasets/` directory.

- BCI Dataset: [https://bupt-ai-cz.github.io/BCI/](URL)
- LLVIP Dataset: [https://drive.google.com/file/d/1VTlT3Y7e1h-Zsne4zahjx5q0TK2ClMVv/view](URL)
- LOLv1 Dataset: [https://drive.google.com/file/d/1L-kqSQyrmMueBh_ziWoPFhfsAh50h20H/view](URL)
- LOLv2 Dataset: [https://drive.google.com/file/d/1Ou9EljYZW8o5dbDCf9R34FS8Pd8kEp2U/view](URL)
- SID Dataset: [https://drive.google.com/drive/folders/1eQ-5Z303sbASEvsgCBSDbhijzLTWQJtR](URL)
- IRVI Dataset: [https://drive.google.com/file/d/1ZcJ0EfF5n_uqtsLc7-8hJgTcr2zHSXY3/view](URL)

## Usage

### 1. Training

To train a model on a specific dataset, use the corresponding `script.py` inside the dataset's folder.

For example, to train the BCI model:
```bash
cd ..
CUDA_VISIBLE_DEVICES=0 python -m bci.script
```

### 2. Evaluation (Metrics)

To evaluate the model and calculate PSNR/SSIM metrics, use the `sampling_and_metric.py` script.

Example: Evaluate the BCI model:
```bash
cd ..
CUDA_VISIBLE_DEVICES=0 python -m bci.sampling_and_metric
```

Ensure the dataset is correctly placed, and pre-trained checkpoints are available before running the evaluation.

### Results

Evaluation metrics (PSNR, SSIM) and generated images are saved in the corresponding output folder.

For any issues, feel free to open an issue on this repository.
