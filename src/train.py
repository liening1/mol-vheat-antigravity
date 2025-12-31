import argparse
import os
import sys
import json
import csv
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, mean_squared_error
import numpy as np

# Add project root to sys.path to allow importing from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset import MoleculeNetDataset
from src.models.mol_vheat import MolVHeat
from src.utils.transforms import get_transforms


def get_scheduler(optimizer, args, warmup_epochs=5):
    """Cosine annealing with linear warmup."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (args.epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/{args.dataset}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Save hyperparameters
    with open(f"{log_dir}/config.json", 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # CSV logger
    csv_file = open(f"{log_dir}/training_log.csv", 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['epoch', 'train_loss', 'val_metric', 'lr', 'is_best'])
    
    # Dataset
    train_transform = get_transforms('train')
    val_transform = get_transforms('val')
    
    train_dataset = MoleculeNetDataset(args.dataset, split='train', root='data', transform=train_transform)
    val_dataset = MoleculeNetDataset(args.dataset, split='val', root='data', transform=val_transform)
    test_dataset = MoleculeNetDataset(args.dataset, split='test', root='data', transform=val_transform)
    
    print(f"Dataset: {args.dataset}")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Model
    task_type = 'classification' if args.dataset == 'bbbp' else 'regression'
    model = MolVHeat(task_type=task_type).to(device)
    
    # Optimizer with warmup
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_scheduler(optimizer, args, warmup_epochs=args.warmup_epochs)
    
    # Loss
    if task_type == 'classification':
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.MSELoss()
        
    best_val_metric = -float('inf') if task_type == 'classification' else float('inf')
    
    print(f"\n{'='*50}")
    print(f"Starting training: {args.epochs} epochs, LR={args.lr}, BS={args.batch_size}")
    print(f"{'='*50}\n")
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False):
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            
        scheduler.step()
        train_loss /= len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Validation
        val_metric = evaluate(model, val_loader, task_type, device)
        
        # Check if best
        is_best = False
        if task_type == 'classification':
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                is_best = True
        else:
            if val_metric < best_val_metric:
                best_val_metric = val_metric
                is_best = True
        
        # Log to CSV
        csv_writer.writerow([epoch+1, f"{train_loss:.6f}", f"{val_metric:.6f}", f"{current_lr:.8f}", is_best])
        csv_file.flush()
        
        # Print progress
        metric_name = "ROC-AUC" if task_type == 'classification' else "RMSE"
        best_marker = " *BEST*" if is_best else ""
        print(f"Epoch {epoch+1:3d}/{args.epochs} | Loss: {train_loss:.4f} | Val {metric_name}: {val_metric:.4f} | LR: {current_lr:.6f}{best_marker}")
                
        if is_best:
            torch.save(model.state_dict(), f"checkpoints/{args.dataset}_best.pth")
            torch.save(model.state_dict(), f"{log_dir}/best_model.pth")
            
    csv_file.close()
    
    # Test
    print(f"\n{'='*50}")
    print("Evaluating on test set...")
    model.load_state_dict(torch.load(f"checkpoints/{args.dataset}_best.pth", weights_only=True))
    test_metric = evaluate(model, test_loader, task_type, device)
    
    metric_name = "ROC-AUC" if task_type == 'classification' else "RMSE"
    print(f"Test {metric_name}: {test_metric:.4f}")
    print(f"{'='*50}")
    
    # Save final results
    results = {
        'dataset': args.dataset,
        'test_metric': test_metric,
        'best_val_metric': best_val_metric,
        'metric_name': metric_name,
        'config': vars(args)
    }
    with open(f"{log_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nLogs saved to: {log_dir}")
    return test_metric


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
    parser = argparse.ArgumentParser(description='Mol-vHeat Training')
    parser.add_argument('--dataset', type=str, default='esol', choices=['esol', 'bbbp'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate (default: 5e-5)')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    args = parser.parse_args()
    
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    train(args)
