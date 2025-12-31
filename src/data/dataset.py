import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image
import requests
import hashlib


class MoleculeNetDataset(Dataset):
    """
    PyTorch Dataset for MoleculeNet datasets.
    Supports: ESOL, BBBP, Lipophilicity, FreeSolv, Tox21, ClinTox
    """
    URLS = {
        'esol': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv',
        'bbbp': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv',
        'lipophilicity': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv',
        'freesolv': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv',
        'tox21': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz',
        'clintox': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz',
    }
    
    # Property column names for each dataset
    PROPERTIES = {
        'esol': 'measured log solubility in mols per litre',
        'bbbp': 'p_np',
        'lipophilicity': 'exp',
        'freesolv': 'expt',
        'tox21': ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 
                  'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'],
        'clintox': ['FDA_APPROVED', 'CT_TOX'],
    }
    
    # Task type for each dataset
    TASK_TYPES = {
        'esol': 'regression',
        'bbbp': 'classification',
        'lipophilicity': 'regression',
        'freesolv': 'regression',
        'tox21': 'multi_classification',
        'clintox': 'multi_classification',
    }

    def __init__(self, name, root='data', split='train', transform=None, img_size=224):
        self.name = name.lower()
        self.root = root
        self.transform = transform
        self.img_size = img_size
        
        if self.name not in self.URLS:
            raise ValueError(f"Dataset {self.name} not supported. Choose from {list(self.URLS.keys())}")
        
        self.task_type = self.TASK_TYPES[self.name]
        self.property_col = self.PROPERTIES[self.name]
        
        self.raw_dir = os.path.join(root, 'raw')
        self.img_dir = os.path.join(root, 'images', self.name)
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.img_dir, exist_ok=True)
        
        # Download and load data
        is_gzip = self.URLS[self.name].endswith('.gz')
        ext = '.csv.gz' if is_gzip else '.csv'
        self.csv_path = os.path.join(self.raw_dir, f'{self.name}{ext}')
        
        if not os.path.exists(self.csv_path):
            self._download()
            
        self.data = pd.read_csv(self.csv_path)
        
        # Filter valid SMILES
        self.data['valid'] = self.data['smiles'].apply(lambda x: Chem.MolFromSmiles(str(x)) is not None)
        self.data = self.data[self.data['valid']]
        
        # For multi-task, filter rows with at least one valid label
        if isinstance(self.property_col, list):
            self.data = self.data.dropna(subset=self.property_col, how='all')
        
        # Deterministic split
        if split:
            np.random.seed(42)
            perm = np.random.permutation(len(self.data))
            train_size = int(0.8 * len(self.data))
            val_size = int(0.1 * len(self.data))
            
            if split == 'train':
                indices = perm[:train_size]
            elif split == 'val':
                indices = perm[train_size:train_size+val_size]
            elif split == 'test':
                indices = perm[train_size+val_size:]
            else:
                indices = np.arange(len(self.data))
                
            self.data = self.data.iloc[indices].reset_index(drop=True)

    def _download(self):
        print(f"Downloading {self.name} dataset...")
        response = requests.get(self.URLS[self.name])
        with open(self.csv_path, 'wb') as f:
            f.write(response.content)
            
    def _smiles_to_image(self, smiles):
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        img = Draw.MolToImage(mol, size=(self.img_size, self.img_size), kekulize=True, fitImage=True)
        return img

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        smiles = row['smiles']
        
        # Get target(s)
        if isinstance(self.property_col, list):
            # Multi-task: return tensor of all labels (NaN -> -1 for masking)
            targets = [row[col] if pd.notna(row[col]) else -1 for col in self.property_col]
            target = torch.tensor(targets, dtype=torch.float32)
        else:
            target = torch.tensor(row[self.property_col], dtype=torch.float32)
        
        # Generate or load image
        img_hash = hashlib.md5(str(smiles).encode()).hexdigest()
        img_path = os.path.join(self.img_dir, f"{img_hash}.png")
        
        if os.path.exists(img_path):
            image = Image.open(img_path).convert('RGB')
        else:
            image = self._smiles_to_image(smiles)
            if image is not None:
                image.save(img_path)
            else:
                image = Image.new('RGB', (self.img_size, self.img_size), 'white')
            
        if self.transform:
            image = self.transform(image)
            
        return image, target


if __name__ == '__main__':
    for ds_name in ['esol', 'bbbp', 'lipophilicity', 'freesolv']:
        try:
            ds = MoleculeNetDataset(ds_name, split='train')
            print(f"{ds_name.upper()}: {len(ds)} samples, task={ds.task_type}")
        except Exception as e:
            print(f"{ds_name.upper()}: Error - {e}")
