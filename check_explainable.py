
import os
import sys
import torch

# Add current directory to path to allow imports from src
sys.path.append(os.getcwd())

print("Testing Explainable-Brain-MRI project structure...")

# Check critical folders
dirs = ["src", "data", "notebooks"]
missing_dirs = []
for d in dirs:
    if os.path.exists(d) and os.path.isdir(d):
        print(f"  [OK] Directory '{d}' found.")
    else:
        print(f"  [MISSING] Directory '{d}' not found.")
        missing_dirs.append(d)

# Check critical modules in src
try:
    from src import model, dataset, gradcam
    print("SUCCESS: Imported 'model', 'dataset', 'gradcam' from 'src'.")
except ImportError as e:
    print(f"FAILURE: Import error from 'src'. Details: {e}")

# Check model file if exists (optional but good)
dataset_path = "best_brain_tumor_model.pth" 
if os.path.exists(dataset_path):
    print(f"  [OK] Model file '{dataset_path}' found.")
else:
    print(f"  [INFO] Model file '{dataset_path}' not found (might be optional).")

print("\nEnvironment info:")
print(f"  Python: {sys.version.split()[0]}")
print(f"  Torch: {torch.__version__}")
