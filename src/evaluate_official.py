"""
Official MoleculeNet Evaluation Script

Uses the same scaffold split as published benchmarks for fair comparison.
Scaffold split groups molecules by their core structure (Bemis-Murcko scaffold).

This ensures:
- Structurally similar molecules are in the same split
- Test set contains truly novel scaffolds
- Results are comparable to published GNN methods
"""

import argparse
import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.mol_vheat import MolVHeat
from src.utils.transforms import get_transforms


def get_scaffold(smiles):
    """Extract Bemis-Murcko scaffold from SMILES."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except:
        return None


def scaffold_split(smiles_list, frac_train=0.8, frac_val=0.1, frac_test=0.1, seed=42):
    """
    Official MoleculeNet scaffold split.
    
    Groups molecules by scaffold, then assigns entire scaffold groups to splits.
    This ensures test set has truly novel molecular structures.
    """
    np.random.seed(seed)
    
    # Get scaffold for each molecule
    scaffolds = {}
    for idx, smiles in enumerate(smiles_list):
        scaffold = get_scaffold(smiles)
        if scaffold is None:
            scaffold = smiles  # Use original SMILES as fallback
        if scaffold not in scaffolds:
            scaffolds[scaffold] = []
        scaffolds[scaffold].append(idx)
    
    # Sort scaffolds by size (largest first for balanced splits)
    scaffold_sets = list(scaffolds.values())
    scaffold_sets.sort(key=len, reverse=True)
    
    # Assign scaffolds to splits
    train_idx, val_idx, test_idx = [], [], []
    train_size = frac_train * len(smiles_list)
    val_size = frac_val * len(smiles_list)
    
    for scaffold_set in scaffold_sets:
        if len(train_idx) < train_size:
            train_idx.extend(scaffold_set)
        elif len(val_idx) < val_size:
            val_idx.extend(scaffold_set)
        else:
            test_idx.extend(scaffold_set)
    
    return train_idx, val_idx, test_idx


class ScaffoldSplitDataset(Dataset):
    """Dataset with official MoleculeNet scaffold split."""
    
    def __init__(self, name, split='train', transform=None, img_size=224, root='data'):
        import pandas as pd
        from rdkit import Chem
        from rdkit.Chem import Draw
        from PIL import Image
        import hashlib
        import requests
        
        self.name = name.lower()
        self.transform = transform
        self.img_size = img_size
        
        # Dataset URLs
        URLS = {
            'esol': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv',
            'bbbp': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv',
            'lipophilicity': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv',
            'freesolv': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv',
        }
        
        PROPERTIES = {
            'esol': 'measured log solubility in mols per litre',
            'bbbp': 'p_np',
            'lipophilicity': 'exp',
            'freesolv': 'expt',
        }
        
        self.property_col = PROPERTIES[self.name]
        self.task_type = 'classification' if self.name == 'bbbp' else 'regression'
        
        # Paths
        raw_dir = os.path.join(root, 'raw')
        self.img_dir = os.path.join(root, 'images', self.name)
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(self.img_dir, exist_ok=True)
        
        csv_path = os.path.join(raw_dir, f'{self.name}.csv')
        
        # Download if needed
        if not os.path.exists(csv_path):
            print(f"Downloading {self.name} dataset...")
            response = requests.get(URLS[self.name])
            with open(csv_path, 'wb') as f:
                f.write(response.content)
        
        # Load data
        data = pd.read_csv(csv_path)
        
        # Filter valid SMILES
        data['valid'] = data['smiles'].apply(lambda x: Chem.MolFromSmiles(str(x)) is not None)
        data = data[data['valid']].reset_index(drop=True)
        
        # Scaffold split
        smiles_list = data['smiles'].tolist()
        train_idx, val_idx, test_idx = scaffold_split(smiles_list)
        
        if split == 'train':
            indices = train_idx
        elif split == 'val':
            indices = val_idx
        elif split == 'test':
            indices = test_idx
        else:
            indices = list(range(len(data)))
        
        self.data = data.iloc[indices].reset_index(drop=True)
        
        # Store for image generation
        self._Chem = Chem
        self._Draw = Draw
        self._Image = Image
        self._hashlib = hashlib
        
        print(f"Scaffold split - {split}: {len(self.data)} samples")
    
    def _smiles_to_image(self, smiles):
        mol = self._Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        img = self._Draw.MolToImage(mol, size=(self.img_size, self.img_size))
        return img
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        smiles = row['smiles']
        target = torch.tensor(row[self.property_col], dtype=torch.float32)
        
        # Get or generate image
        img_hash = self._hashlib.md5(str(smiles).encode()).hexdigest()
        img_path = os.path.join(self.img_dir, f"{img_hash}.png")
        
        if os.path.exists(img_path):
            image = self._Image.open(img_path).convert('RGB')
        else:
            image = self._smiles_to_image(smiles)
            if image is not None:
                image.save(img_path)
            else:
                image = self._Image.new('RGB', (self.img_size, self.img_size), 'white')
        
        if self.transform:
            image = self.transform(image)
        
        return image, target


def evaluate_official(model_path, dataset_name, device=None):
    """Run official MoleculeNet evaluation with scaffold split."""
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*60)
    print("Official MoleculeNet Evaluation (Scaffold Split)")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_name}")
    print(f"Device: {device}")
    print()
    
    # Load data with scaffold split
    transform = get_transforms('val')
    
    train_dataset = ScaffoldSplitDataset(dataset_name, 'train', transform)
    val_dataset = ScaffoldSplitDataset(dataset_name, 'val', transform)  
    test_dataset = ScaffoldSplitDataset(dataset_name, 'test', transform)
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # Load model
    task_type = test_dataset.task_type
    model = MolVHeat(task_type=task_type).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    model.eval()
    
    # Evaluate
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            
            if task_type == 'classification':
                preds = torch.sigmoid(outputs)
            else:
                preds = outputs
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
    
    preds = np.concatenate(all_preds).flatten()
    labels = np.concatenate(all_labels).flatten()
    
    # Compute metrics
    if task_type == 'regression':
        rmse = np.sqrt(mean_squared_error(labels, preds))
        mae = mean_absolute_error(labels, preds)
        r2 = r2_score(labels, preds)
        pearson_r, _ = stats.pearsonr(labels, preds)
        
        results = {
            'dataset': dataset_name,
            'split_type': 'scaffold',
            'test_rmse': round(rmse, 4),
            'test_mae': round(mae, 4),
            'test_r2': round(r2, 4),
            'pearson_r': round(pearson_r, 4),
            'n_train': len(train_dataset),
            'n_val': len(val_dataset),
            'n_test': len(test_dataset),
            'timestamp': datetime.now().isoformat()
        }
        
        print("\n📊 Results (Scaffold Split)")
        print("-"*40)
        print(f"   Test RMSE:   {rmse:.4f}")
        print(f"   Test MAE:    {mae:.4f}")
        print(f"   Test R²:     {r2:.4f}")
        print(f"   Pearson r:   {pearson_r:.4f}")
    else:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(labels, preds)
        
        results = {
            'dataset': dataset_name,
            'split_type': 'scaffold',
            'test_roc_auc': round(auc, 4),
            'n_train': len(train_dataset),
            'n_val': len(val_dataset),
            'n_test': len(test_dataset),
            'timestamp': datetime.now().isoformat()
        }
        
        print("\n📊 Results (Scaffold Split)")
        print("-"*40)
        print(f"   Test ROC-AUC: {auc:.4f}")
    
    print("-"*40)
    print(f"   Train/Val/Test: {len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}")
    print("="*60)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Official MoleculeNet Evaluation')
    parser.add_argument('--model', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--dataset', type=str, default='esol', 
                        choices=['esol', 'bbbp', 'lipophilicity', 'freesolv'])
    parser.add_argument('--output', type=str, default='evaluation_results',
                        help='Output directory')
    args = parser.parse_args()
    
    results = evaluate_official(args.model, args.dataset)
    
    # Save results
    os.makedirs(args.output, exist_ok=True)
    output_path = os.path.join(args.output, f"official_{args.dataset}_scaffold.json")
    
    def _json_default(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=_json_default)
    
    print(f"\nResults saved to: {output_path}")
    return results


if __name__ == '__main__':
    main()
