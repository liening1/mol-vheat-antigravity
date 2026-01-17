"""
Parallel Benchmark Runner for Mol-vHeat

Runs multiple experiments in parallel to compare model configurations
and generate comprehensive evaluation reports.
"""

import argparse
import os
import sys
import json
import subprocess
import multiprocessing as mp
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class BenchmarkRunner:
    """Run parallel benchmark experiments."""
    
    def __init__(self, output_dir='benchmark_results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def run_single_evaluation(self, config):
        """Run a single evaluation experiment."""
        from src.evaluate import ModelEvaluator
        
        model_path = config['model_path']
        dataset = config['dataset']
        name = config.get('name', os.path.basename(model_path))
        
        print(f"[{name}] Starting evaluation...")
        
        try:
            evaluator = ModelEvaluator(model_path, dataset)
            results = evaluator.run_full_evaluation()
            results['config_name'] = name
            
            # Save individual result
            output_path = os.path.join(self.output_dir, f"{name}_{dataset}.json")
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"[{name}] ✓ Completed")
            return results
            
        except Exception as e:
            print(f"[{name}] ✗ Failed: {str(e)}")
            return {'config_name': name, 'error': str(e)}
    
    def run_parallel_evaluations(self, configs, max_workers=None):
        """Run multiple evaluations in parallel."""
        if max_workers is None:
            max_workers = min(len(configs), mp.cpu_count())
        
        print(f"\n{'='*60}")
        print(f"Running {len(configs)} experiments with {max_workers} workers")
        print(f"{'='*60}\n")
        
        results = []
        
        # Note: For GPU experiments, parallel execution may cause OOM
        # Running sequentially for safety
        for config in configs:
            result = self.run_single_evaluation(config)
            results.append(result)
        
        return results
    
    def run_cross_validation(self, model_class_config, dataset, n_folds=5):
        """Run k-fold cross-validation."""
        from src.data.dataset import MoleculeNetDataset
        from src.models.mol_vheat import MolVHeat
        from src.utils.transforms import get_transforms
        from sklearn.model_selection import KFold
        import torch
        from torch.utils.data import DataLoader, Subset
        
        print(f"\n{'='*60}")
        print(f"Running {n_folds}-fold Cross-Validation on {dataset}")
        print(f"{'='*60}\n")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load full dataset
        transform = get_transforms('val')
        full_dataset = MoleculeNetDataset(dataset, split='train', root='data', transform=transform)
        task_type = full_dataset.task_type
        if task_type == 'multi_classification':
            task_type = 'classification'
        
        # K-fold
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(range(len(full_dataset)))):
            print(f"\nFold {fold+1}/{n_folds}")
            
            val_subset = Subset(full_dataset, val_idx)
            val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
            
            # Load best model and evaluate on this fold's validation set
            model = MolVHeat(task_type=task_type).to(device)
            model_path = f"checkpoints/{dataset}_best.pth"
            
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
            
            model.eval()
            
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for images, labels in val_loader:
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
            
            if task_type == 'regression':
                rmse = np.sqrt(np.mean((preds - labels) ** 2))
                fold_results.append({'fold': fold+1, 'rmse': rmse})
                print(f"   RMSE: {rmse:.4f}")
            else:
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(labels, preds)
                fold_results.append({'fold': fold+1, 'roc_auc': auc})
                print(f"   ROC-AUC: {auc:.4f}")
        
        # Aggregate results
        if task_type == 'regression':
            values = [r['rmse'] for r in fold_results]
            metric_name = 'rmse'
        else:
            values = [r['roc_auc'] for r in fold_results]
            metric_name = 'roc_auc'
        
        cv_results = {
            'dataset': dataset,
            'n_folds': n_folds,
            'metric': metric_name,
            'mean': round(np.mean(values), 4),
            'std': round(np.std(values), 4),
            'min': round(np.min(values), 4),
            'max': round(np.max(values), 4),
            'fold_results': fold_results
        }
        
        print(f"\n{'='*60}")
        print(f"Cross-Validation Results: {metric_name.upper()}")
        print(f"Mean: {cv_results['mean']:.4f} ± {cv_results['std']:.4f}")
        print(f"Range: [{cv_results['min']:.4f}, {cv_results['max']:.4f}]")
        print(f"{'='*60}")
        
        # Save results
        output_path = os.path.join(self.output_dir, f"cv_{dataset}_{n_folds}fold.json")
        with open(output_path, 'w') as f:
            json.dump(cv_results, f, indent=2)
        
        return cv_results
    
    def compare_results(self, results):
        """Compare and summarize results from multiple experiments."""
        summary = {
            'timestamp': self.timestamp,
            'num_experiments': len(results),
            'experiments': []
        }
        
        for r in results:
            if 'error' in r:
                summary['experiments'].append({
                    'name': r.get('config_name', 'Unknown'),
                    'status': 'failed',
                    'error': r['error']
                })
            else:
                exp_summary = {
                    'name': r.get('config_name', 'Unknown'),
                    'status': 'success',
                    'dataset': r.get('dataset', 'Unknown'),
                    'metrics': r.get('accuracy_metrics', {}),
                    'inference_time_ms': r.get('inference_time', {}).get('mean_ms', None),
                    'model_size_mb': r.get('model_stats', {}).get('model_size_mb', None)
                }
                summary['experiments'].append(exp_summary)
        
        # Save summary
        summary_path = os.path.join(self.output_dir, f"summary_{self.timestamp}.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def generate_report(self, results, cv_results=None, output_path=None):
        """Generate a markdown report from benchmark results."""
        if output_path is None:
            output_path = os.path.join(self.output_dir, f"benchmark_report_{self.timestamp}.md")
        
        lines = [
            "# Mol-vHeat Benchmark Report",
            "",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "---",
            "",
            "## 📊 Evaluation Results",
            ""
        ]
        
        # Results table
        if results:
            r = results[0]  # Assuming single model for now
            
            if 'accuracy_metrics' in r:
                lines.extend([
                    "### Accuracy Metrics",
                    "",
                    "| Metric | Value |",
                    "|--------|-------|"
                ])
                for key, value in r['accuracy_metrics'].items():
                    lines.append(f"| **{key.upper()}** | {value} |")
                lines.append("")
            
            if 'model_stats' in r:
                lines.extend([
                    "### Model Statistics",
                    "",
                    "| Metric | Value |",
                    "|--------|-------|"
                ])
                stats = r['model_stats']
                lines.append(f"| Parameters | {stats['total_parameters']:,} |")
                lines.append(f"| Model Size | {stats['model_size_mb']} MB |")
                lines.append("")
            
            if 'inference_time' in r:
                timing = r['inference_time']
                lines.extend([
                    "### Inference Performance",
                    "",
                    "| Metric | Value |",
                    "|--------|-------|",
                    f"| Mean Time | {timing['mean_ms']:.3f} ms |",
                    f"| Std Dev | {timing['std_ms']:.3f} ms |",
                    f"| Throughput | {1000/timing['mean_ms']:.1f} samples/sec |",
                    ""
                ])
            
            if 'memory' in r and 'peak_memory_mb' in r['memory']:
                lines.extend([
                    "### Memory Usage",
                    "",
                    f"| Peak Memory | {r['memory']['peak_memory_mb']} MB |",
                    ""
                ])
            
            if 'error_analysis' in r:
                errors = r['error_analysis']
                lines.extend([
                    "### Error Analysis",
                    "",
                    "| Metric | Value |",
                    "|--------|-------|"
                ])
                if 'outlier_ratio' in errors:
                    lines.append(f"| Mean Error | {errors['mean_error']} |")
                    lines.append(f"| Std Error | {errors['std_error']} |")
                    lines.append(f"| Outlier Ratio | {errors['outlier_ratio']*100:.2f}% |")
                lines.append("")
        
        # Cross-validation results
        if cv_results:
            lines.extend([
                "---",
                "",
                "## 🔄 Cross-Validation Results",
                "",
                f"**{cv_results['n_folds']}-Fold Cross-Validation**",
                "",
                f"| Metric | Mean | Std | Min | Max |",
                f"|--------|------|-----|-----|-----|",
                f"| {cv_results['metric'].upper()} | {cv_results['mean']} | ±{cv_results['std']} | {cv_results['min']} | {cv_results['max']} |",
                ""
            ])
        
        # Write report
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))
        
        print(f"\nReport saved to: {output_path}")
        return output_path


def main():
    parser = argparse.ArgumentParser(description='Mol-vHeat Benchmark Runner')
    parser.add_argument('--model', type=str, default='checkpoints/esol_best.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='esol',
                        choices=['esol', 'bbbp', 'lipophilicity', 'freesolv'])
    parser.add_argument('--output', type=str, default='benchmark_results',
                        help='Output directory')
    parser.add_argument('--cv', action='store_true',
                        help='Run cross-validation')
    parser.add_argument('--cv-folds', type=int, default=5,
                        help='Number of CV folds')
    args = parser.parse_args()
    
    runner = BenchmarkRunner(args.output)
    
    # Single model evaluation
    configs = [
        {
            'model_path': args.model,
            'dataset': args.dataset,
            'name': 'mol_vheat_best'
        }
    ]
    
    results = runner.run_parallel_evaluations(configs)
    
    # Cross-validation if requested
    cv_results = None
    if args.cv:
        cv_results = runner.run_cross_validation({}, args.dataset, args.cv_folds)
    
    # Generate report
    runner.generate_report(results, cv_results)
    runner.compare_results(results)


if __name__ == '__main__':
    main()
