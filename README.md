# Mol-vHeat Implementation Walkthrough

I have implemented the Mol-vHeat framework for molecular property prediction using the vHeat backbone.

## Implemented Components

### 1. Data Pipeline (`src/data/`)
- **`dataset.py`**: `MoleculeNetDataset` class that automatically downloads ESOL/BBBP datasets, converts SMILES strings to 2D images using RDKit, and handles caching.
- **`transforms.py`**: Implements `RandomRotation` (0-360 degrees) for data augmentation, as molecular properties are rotation-invariant.

### 2. Model Architecture (`src/models/`)
- **`vheat.py`**: The official vHeat backbone implementation.
- **`mol_vheat.py`**: A wrapper class `MolVHeat` that adapts the backbone for regression (ESOL) or classification (BBBP) tasks. It replaces the original classifier with a task-specific head.

### 3. Training (`src/train.py`)
- A complete training script supporting:
    - Automatic device selection (CUDA/CPU).
    - AdamW optimizer with Cosine Annealing scheduler.
    - MSE Loss for regression, BCEWithLogits Loss for classification.
    - Model checkpointing (saves best model based on validation metric).

## Usage

To train the model on ESOL (Regression):
```bash
python src/train.py --dataset esol --epochs 100 --batch_size 32
```

To train the model on BBBP (Classification):
```bash
python src/train.py --dataset bbbp --epochs 100 --batch_size 32
```

## Environment Note
> [!WARNING]
> **Python Version Incompatibility**
> The current system environment is running **Python 3.13.5**, which is not yet supported by PyTorch (as of late 2024/early 2025).
> 
> **Action Required:**
> Please run this project in an environment with **Python 3.8 - 3.11**.
> You can create a conda environment:
> ```bash
> conda create -n molvheat python=3.10
> conda activate molvheat
> ```

## Deployment on HPC

To run this project on a High Performance Computing (HPC) cluster:

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/liening1/mol-vheat-antigravity.git
    cd mol-vheat-antigravity
    ```

2.  **Set up Environment**:
    Most HPCs use `module` or `conda`. We recommend `conda`.
    ```bash
    # Load conda module if needed (e.g., module load anaconda3)
    conda create -n molvheat python=3.10 -y
    conda activate molvheat
    pip install -r requirements.txt
    ```

3.  **Run Training**:
    It is best to submit a job script (e.g., SLURM).
    
    *Example `run_job.sh` (SLURM)*:
    ```bash
    #!/bin/bash
    #SBATCH --job-name=molvheat
    #SBATCH --output=logs/%j.out
    #SBATCH --gres=gpu:1
    #SBATCH --time=4:00:00
    
    source activate molvheat
    python src/train.py --dataset esol
    ```
