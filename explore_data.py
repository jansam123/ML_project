#!/usr/bin/env python3
"""
Script to load and explore a ROOT file using uproot.
"""

import uproot
import matplotlib.pyplot as plt
import numpy as np

# Load the ROOT file
root_file_path = "/home/scur0034/ML_project/data/JetClass/Pythia/train_100M/HToBB_000.root"
print(f"Loading ROOT file: {root_file_path}")

root_file = uproot.open(root_file_path)
print("\n✓ File loaded successfully!\n")

# Print available keys
print("Available keys in the ROOT file:")
print("-" * 50)
keys = root_file.keys()
for i, key in enumerate(keys, 1):
    print(f"{i}. {key}")
print("-" * 50)

first_key = 'tree'
print(f"\nExploring '{first_key}'...")

obj = root_file[first_key]
print(f"Type: {type(obj)}")
    

branches = obj.keys()
print(f"\nAvailable branches:")
for i, branch in enumerate(branches, 1):
    print(f"  {i}. {branch}")


first_branch = 'jet_pt'
print(f"\nLoading data from branch '{first_branch}'...")

data = obj[first_branch].array(library="np")

# Create a simple histogram plot
plt.figure(figsize=(10, 6))
plt.hist(data, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel(first_branch)
plt.ylabel("Frequency")
plt.title(f"Histogram of {first_branch}")
plt.tight_layout()

# Save and show the plot
#output_path = "logs/plot_exploration.png"
#plt.savefig(output_path, dpi=100)
#print(f"✓ Plot saved to: {output_path}")
plt.show()

# Print some basic statistics
print(f"\nBasic statistics for '{first_branch}':")
print(f"  Mean: {np.mean(data):.4f}")
print(f"  Std: {np.std(data):.4f}")
print(f"  Min: {np.min(data):.4f}")
print(f"  Max: {np.max(data):.4f}")
print(f"  Total entries: {len(data)}")

#print(data)