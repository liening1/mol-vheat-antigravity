"""
Comprehensive Model Evaluation Script for Mol-vHeat

Evaluates model performance across multiple dimensions:
- Accuracy metrics: RMSE, MAE, R², Pearson, Spearman
- Computational efficiency: Inference time, memory usage, model size
- Quality indices: Error distribution, outlier analysis
"""

import argparse
import os
import sys
import json
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy import stats
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score,
    roc_auc_score,
    accuracy_score,
    precision_recall_fscore_support
)

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset import MoleculeNetDataset
from src.models.mol_vheat import MolVHeat
from src.utils.transforms import get_transforms


def get_device_info():
    """Get detailed GPU/CPU device information."""
    info = {
        'device_type': 'cpu',
        'device_name': 'CPU',
        'cuda_version': None,
        'gpu_memory_total': None
    }
    
    if torch.cuda.is_available():
        info['device_type'] = 'cuda'
        info['device_name'] = torch.cuda.get_device_name(0)
        info['cuda_version'] = torch.version.cuda
        info['gpu_memory_total'] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
    
    return info


def print_device_info():
    """Print device information at job end."""
    info = get_device_info()
    print(f"\n{'='*50}")
    print("Hardware Information")
    print(f"{'='*50}")
    print(f"   Device: {info['device_name']}")
    if info['cuda_version']:
        print(f"   CUDA Version: {info['cuda_version']}")
        print(f"   GPU Memory: {info['gpu_memory_total']}")
    print(f"{'='*50}")


class ModelEvaluator:
    """Comprehensive model evaluator for Mol-vHeat."""
    
    def __init__(self, model_path, dataset_name, device=None, background: str = 'none'):
        self.model_path = model_path
        self.dataset_name = dataset_name
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load dataset
        self.transform = get_transforms('val', background=background)
        self.test_dataset = MoleculeNetDataset(
            dataset_name, split='test', root='data', transform=self.transform
        )
        self.task_type = self.test_dataset.task_type
        if self.task_type == 'multi_classification':
            self.task_type = 'classification'
            
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=32, shuffle=False, num_workers=4
        )
        
        # Load model
        self.model = MolVHeat(task_type=self.task_type).to(self.device)
        state_dict = torch.load(model_path, weights_only=True, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        # Results storage
        self.results = {
            'model_path': model_path,
            'dataset': dataset_name,
            'task_type': self.task_type,
            'timestamp': datetime.now().isoformat(),
            'device': str(self.device),
            'hardware_info': get_device_info()
        }
        
    def count_parameters(self):
        """Count model parameters."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Model file size
        model_size_mb = os.path.getsize(self.model_path) / (1024 * 1024)
        
        self.results['model_stats'] = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': round(model_size_mb, 2)
        }
        return self.results['model_stats']
    
    def measure_inference_time(self, num_runs=100, warmup_runs=10):
        """Measure inference time per sample."""
        # Create dummy input
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        
        # Warmup
        for _ in range(warmup_runs):
            with torch.no_grad():
                _ = self.model(dummy_input)
        
        # Synchronize if using CUDA
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Measure
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            with torch.no_grad():
                _ = self.model(dummy_input)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
        
        self.results['inference_time'] = {
            'mean_ms': round(np.mean(times), 3),
            'std_ms': round(np.std(times), 3),
            'min_ms': round(np.min(times), 3),
            'max_ms': round(np.max(times), 3),
            'num_runs': num_runs
        }
        return self.results['inference_time']
    
    def measure_memory(self):
        """Measure GPU memory usage."""
        if self.device.type != 'cuda':
            self.results['memory'] = {'note': 'GPU memory measurement requires CUDA'}
            return self.results['memory']
        
        torch.cuda.reset_peak_memory_stats()
        
        # Forward pass with batch
        dummy_input = torch.randn(32, 3, 224, 224).to(self.device)
        with torch.no_grad():
            _ = self.model(dummy_input)
        
        peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
        current_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
        
        self.results['memory'] = {
            'peak_memory_mb': round(peak_memory, 2),
            'current_memory_mb': round(current_memory, 2),
            'batch_size': 32
        }
        return self.results['memory']
    
    def get_predictions(self):
        """Get all predictions and labels."""
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                
                if self.task_type == 'classification':
                    preds = torch.sigmoid(outputs)
                else:
                    preds = outputs
                
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.numpy())
        
        return np.concatenate(all_preds).flatten(), np.concatenate(all_labels).flatten()
    
    def compute_accuracy_metrics(self):
        """Compute all accuracy metrics."""
        preds, labels = self.get_predictions()
        
        if self.task_type == 'regression':
            # Regression metrics
            rmse = np.sqrt(mean_squared_error(labels, preds))
            mae = mean_absolute_error(labels, preds)
            r2 = r2_score(labels, preds)
            pearson_r, pearson_p = stats.pearsonr(labels, preds)
            spearman_r, spearman_p = stats.spearmanr(labels, preds)
            
            self.results['accuracy_metrics'] = {
                'rmse': round(rmse, 4),
                'mae': round(mae, 4),
                'r2': round(r2, 4),
                'pearson_r': round(pearson_r, 4),
                'pearson_p_value': round(pearson_p, 6),
                'spearman_r': round(spearman_r, 4),
                'spearman_p_value': round(spearman_p, 6)
            }
        else:
            # Classification metrics
            roc_auc = roc_auc_score(labels, preds)
            binary_preds = (preds > 0.5).astype(int)
            accuracy = accuracy_score(labels, binary_preds)
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, binary_preds, average='binary', zero_division=0
            )
            
            self.results['accuracy_metrics'] = {
                'roc_auc': round(roc_auc, 4),
                'accuracy': round(accuracy, 4),
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1_score': round(f1, 4)
            }
        
        return self.results['accuracy_metrics']
    
    def analyze_errors(self):
        """Analyze error distribution and outliers."""
        preds, labels = self.get_predictions()
        
        if self.task_type == 'regression':
            errors = preds - labels
            abs_errors = np.abs(errors)
            
            # Error statistics
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            
            # Outlier analysis (errors > 2 std)
            outlier_threshold = 2 * std_error
            outliers = np.abs(errors) > outlier_threshold
            outlier_ratio = np.mean(outliers)
            
            # Percentiles
            percentiles = {
                'p50': round(np.percentile(abs_errors, 50), 4),
                'p75': round(np.percentile(abs_errors, 75), 4),
                'p90': round(np.percentile(abs_errors, 90), 4),
                'p95': round(np.percentile(abs_errors, 95), 4),
                'p99': round(np.percentile(abs_errors, 99), 4)
            }
            
            self.results['error_analysis'] = {
                'mean_error': round(mean_error, 4),
                'std_error': round(std_error, 4),
                'outlier_threshold': round(outlier_threshold, 4),
                'outlier_ratio': round(outlier_ratio, 4),
                'num_outliers': int(np.sum(outliers)),
                'total_samples': len(errors),
                'error_percentiles': percentiles
            }
        else:
            # For classification, analyze confidence distribution
            correct = (preds > 0.5) == labels
            confidence = np.where(labels == 1, preds, 1 - preds)
            
            self.results['error_analysis'] = {
                'mean_confidence': round(np.mean(confidence), 4),
                'std_confidence': round(np.std(confidence), 4),
                'low_confidence_ratio': round(np.mean(confidence < 0.6), 4),
                'high_confidence_correct': round(np.mean(correct[confidence > 0.8]), 4)
            }
        
        return self.results['error_analysis']
    
    def compute_flops(self):
        """Estimate FLOPs using a simple forward pass analysis."""
        try:
            from thop import profile
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            flops, params = profile(self.model, inputs=(dummy_input,), verbose=False)
            self.results['flops'] = {
                'total_flops': int(flops),
                'gflops': round(flops / 1e9, 2)
            }
        except ImportError:
            # Estimate based on model architecture
            # vHeat-T has approximately 5.7 GFLOPs
            self.results['flops'] = {
                'estimated_gflops': 5.7,
                'note': 'Estimated (install thop for precise measurement)'
            }
        return self.results['flops']
    
    def run_full_evaluation(self):
        """Run all evaluations."""
        print(f"="*60)
        print(f"Mol-vHeat Comprehensive Evaluation")
        print(f"="*60)
        print(f"Model: {self.model_path}")
        print(f"Dataset: {self.dataset_name}")
        print(f"Device: {self.device}")
        print()
        
        print("📊 Computing model statistics...")
        stats = self.count_parameters()
        print(f"   Parameters: {stats['total_parameters']:,}")
        print(f"   Model size: {stats['model_size_mb']:.2f} MB")
        
        print("\n⏱️  Measuring inference time...")
        timing = self.measure_inference_time()
        print(f"   Mean: {timing['mean_ms']:.3f} ms/sample")
        print(f"   Std:  {timing['std_ms']:.3f} ms")
        
        print("\n💾 Measuring memory usage...")
        memory = self.measure_memory()
        if 'peak_memory_mb' in memory:
            print(f"   Peak memory: {memory['peak_memory_mb']:.2f} MB")
        else:
            print(f"   {memory.get('note', 'N/A')}")
        
        print("\n📈 Computing accuracy metrics...")
        metrics = self.compute_accuracy_metrics()
        for key, value in metrics.items():
            print(f"   {key}: {value}")
        
        print("\n🔍 Analyzing errors...")
        errors = self.analyze_errors()
        if 'outlier_ratio' in errors:
            print(f"   Outlier ratio: {errors['outlier_ratio']*100:.2f}%")
            print(f"   Error std: {errors['std_error']:.4f}")
        
        print("\n⚡ Computing FLOPs...")
        flops = self.compute_flops()
        if 'gflops' in flops:
            print(f"   GFLOPs: {flops['gflops']}")
        else:
            print(f"   Estimated GFLOPs: {flops.get('estimated_gflops', 'N/A')}")
        
        print(f"\n{'='*60}")
        print("Evaluation complete!")
        
        return self.results
    
    def save_results(self, output_dir):
        """Save results to JSON file."""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"evaluation_{self.dataset_name}_{timestamp}.json")

        def _json_default(obj):
            # NumPy scalars/arrays
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            # Torch tensors
            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu().tolist()
            # Fallback
            return str(obj)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=_json_default)
        
        print(f"Results saved to: {output_path}")
        return output_path


def main():
    parser = argparse.ArgumentParser(description='Mol-vHeat Model Evaluation')
    parser.add_argument('--model', type=str, required=True, 
                        help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='esol',
                        choices=['esol', 'bbbp', 'lipophilicity', 'freesolv', 'tox21', 'clintox'])
    parser.add_argument(
        '--background',
        type=str,
        default='none',
        choices=['none', 'mean', 'mean_jitter'],
        help="Background handling (must match training): 'none', 'mean', or 'mean_jitter' (val/test typically use 'mean').",
    )
    parser.add_argument('--output', type=str, default='evaluation_results',
                        help='Output directory for results')
    args = parser.parse_args()
    
    evaluator = ModelEvaluator(args.model, args.dataset, background=args.background)
    results = evaluator.run_full_evaluation()
    evaluator.save_results(args.output)
    
    # Print hardware info at job end
    print_device_info()
    
    return results


if __name__ == '__main__':
    main()
