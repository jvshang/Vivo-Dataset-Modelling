#!/usr/bin/env python3
import os
import sys
import json
import numpy as np
from pathlib import Path

# Add src to sys.path so we can import utils_plot
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils_plot import (
    _log_confusion_matrix,
    _log_roc_curves,
    _log_imbalance_and_performance,
    compare_models
)

def main():
    figures_dir = Path("figures")
    if not figures_dir.exists():
        print("No 'figures' directory found. Run training first to generate data.")
        return

    print("Scanning figures directory for saved plot data...")
    count = 0
    for root, dirs, files in os.walk(figures_dir):
        root_path = Path(root)
        for file in files:
            file_path = root_path / file
            
            # Determine prefix by removing the known data suffix
            # E.g., Task1_USFalse_CWFalse_rf_L0.1_S0.05_data_confusion.npz
            # Prefix = Task1_USFalse_CWFalse_rf_L0.1_S0.05_
            
            if file.endswith("data_confusion.npz"):
                prefix = file.replace("data_confusion.npz", "")
                data = np.load(file_path, allow_pickle=True)
                _log_confusion_matrix(
                    data["y_true"], 
                    data["y_pred"], 
                    data["class_names"].tolist(), 
                    root_path, 
                    prefix=prefix
                )
                print(f" Regenerated confusion matrix: {root_path / f'{prefix}confusion_matrix.png'}")
                count += 1
                
            elif file.endswith("data_roc.npz"):
                prefix = file.replace("data_roc.npz", "")
                data = np.load(file_path, allow_pickle=True)
                _log_roc_curves(
                    data["y_true"], 
                    data["y_proba"], 
                    data["class_names"].tolist(), 
                    root_path, 
                    prefix=prefix
                )
                print(f" Regenerated ROC curves: {root_path / f'{prefix}roc_curves.png'}")
                count += 1
                
            elif file.endswith("data_imbalance.npz"):
                prefix = file.replace("data_imbalance.npz", "")
                data = np.load(file_path, allow_pickle=True)
                _log_imbalance_and_performance(
                    data["y_train"], 
                    data["y_test"], 
                    data["y_pred"], 
                    data["class_names"].tolist(), 
                    root_path, 
                    prefix=prefix
                )
                print(f" Regenerated imbalance & performance plot: {root_path / f'{prefix}imbalance_performance.png'}")
                count += 1
                
            elif file.endswith("data_compare.json"):
                prefix = file.replace("data_compare.json", "")
                with open(file_path, "r") as f:
                    results = json.load(f)
                
                # We need L and S for compare_models, but it's embedded in the results dict too.
                if len(results) > 0:
                    L = results[0].get("L", 0.0)
                    S = results[0].get("S", 0.0)
                else:
                    L, S = 0.0, 0.0
                    
                compare_models(results, L, S, root_path, prefix=prefix)
                print(f" Regenerated model comparison: {root_path / f'{prefix}comparison.png'}")
                count += 1

    print(f"\nDone! Regenerated {count} plots from raw data.")

if __name__ == "__main__":
    main()
