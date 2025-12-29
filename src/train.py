import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data.dataset import MoleculeNetDataset
from src.models.mol_vheat import MolVHeat
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, mean_squared_error
import numpy as np

from src.utils.transforms import get_transforms

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataset
    train_transform = get_transforms('train')
    val_transform = get_transforms('val')
    
    train_dataset = MoleculeNetDataset(args.dataset, split='train', root='data', transform=train_transform)
    val_dataset = MoleculeNetDataset(args.dataset, split='val', root='data', transform=val_transform)
    test_dataset = MoleculeNetDataset(args.dataset, split='test', root='data', transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Model
    task_type = 'classification' if args.dataset == 'bbbp' else 'regression'
    model = MolVHeat(task_type=task_type).to(device)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Loss
    if task_type == 'classification':
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.MSELoss()
        
    best_val_metric = -float('inf') if task_type == 'classification' else float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        scheduler.step()
        train_loss /= len(train_loader)
        
        # Validation
        val_metric = evaluate(model, val_loader, task_type, device)
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Metric={val_metric:.4f}")
        
        # Save best
        is_best = False
        if task_type == 'classification':
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                is_best = True
        else:
            if val_metric < best_val_metric:
                best_val_metric = val_metric
                is_best = True
                
        if is_best:
            torch.save(model.state_dict(), f"checkpoints/{args.dataset}_best.pth")
            print("Saved best model.")
            
    # Test
    model.load_state_dict(torch.load(f"checkpoints/{args.dataset}_best.pth"))
    test_metric = evaluate(model, test_loader, task_type, device)
    print(f"Test Metric: {test_metric:.4f}")

def evaluate(model, loader, task_type, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            if task_type == 'classification':
                preds = torch.sigmoid(outputs)
            else:
                preds = outputs
                
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    if task_type == 'classification':
        return roc_auc_score(all_labels, all_preds)
    else:
        return np.sqrt(mean_squared_error(all_labels, all_preds))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='esol', choices=['esol', 'bbbp'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    
    os.makedirs('checkpoints', exist_ok=True)
    train(args)
