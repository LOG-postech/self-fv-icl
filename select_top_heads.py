#!/usr/bin/env python3
"""
Select salient attention heads from causal-indirect-effect (CIE) outputs.

`main_icl/main.py --cie 1` saves the CIE of a single layer at a time as
`{layer}_mean_effect.pt`. This script merges those per-layer files into a single CIE
tensor and selects the top-k heads, writing `top_heads.json` (and `correct_top_heads.json`)
that the self-function-vector experiments load via `--top_heads_path`.
"""

import argparse
import json
import os
import sys
from glob import glob
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Merge CIE files for specified output directory and number of layers'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        required=True, 
        help='Root output directory containing the CIE results (e.g., results)'
    )
    parser.add_argument(
        '--num_layers', 
        type=int, 
        required=True, 
        help='Number of layers (e.g., 32)'
    )
    parser.add_argument(
        '--top_k', 
        type=int, 
        default=100, 
        help='Number of top attention heads to save (default: 100)'
    )
    return parser.parse_args()


def get_layer_file_paths(directory: Path, layer: int) -> Tuple[Path, Path]:
    """Get file paths for a specific layer."""
    cie_path = directory / f'{layer}_mean_effect.pt'
    correct_cie_path = directory / f'{layer}_correct_mean_effect.pt'
    return cie_path, correct_cie_path


def check_ingredients_exist(directory: Path, num_layers: int) -> bool:
    """Check if all required CIE files exist for all layers."""
    for layer in range(num_layers):
        cie_path, correct_cie_path = get_layer_file_paths(directory, layer)
        if not cie_path.exists() or not correct_cie_path.exists():
            return False
    return True


def is_already_processed(directory: Path) -> bool:
    """Check if directory has already been processed."""
    return (directory / 'cie_merged.pt').exists()


def validate_directories(directories: List[Path], num_layers: int) -> List[Path]:
    """
    Validate directories and return list of directories ready for processing.
    
    Returns:
        List of directories that have all ingredients and aren't already processed.
    """
    valid_directories = []
    
    for directory in directories:
        if is_already_processed(directory):
            print(f"Already processed: {directory}")
            continue
            
        if not check_ingredients_exist(directory, num_layers):
            print(f"Missing ingredients in {directory}, skipping.")
            continue
            
        valid_directories.append(directory)
    
    return valid_directories


def load_and_merge_layers(directory: Path, num_layers: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load and merge CIE tensors across all layers."""
    cie_merged = None
    correct_cie_merged = None
    
    for layer in range(num_layers):
        cie_path, correct_cie_path = get_layer_file_paths(directory, layer)
        
        try:
            cie_layer = torch.load(cie_path, map_location='cpu')
            correct_cie_layer = torch.load(correct_cie_path, map_location='cpu')
        except Exception as e:
            raise RuntimeError(f"Failed to load layer {layer} from {directory}: {e}")
        
        if cie_merged is None:
            cie_merged = cie_layer.clone()
            correct_cie_merged = correct_cie_layer.clone()
        else:
            cie_merged[layer] = cie_layer[layer]
            correct_cie_merged[layer] = correct_cie_layer[layer]
    
    return cie_merged, correct_cie_merged


def compute_top_heads(tensor: torch.Tensor, top_k: int) -> List[Tuple[int, int, float]]:
    """Compute top-k attention heads from merged tensor."""
    topk_vals, topk_inds = torch.topk(tensor.flatten(), k=top_k)
    
    # Convert flat indices back to (layer, head) coordinates
    layer_inds, head_inds = np.unravel_index(
        topk_inds.cpu().numpy(), 
        tensor.shape
    )
    
    return [
        (int(layer), int(head), float(effect))
        for layer, head, effect in zip(
            layer_inds, 
            head_inds, 
            topk_vals.cpu().numpy()
        )
    ]


def save_results(directory: Path, cie_merged: torch.Tensor, 
                correct_cie_merged: torch.Tensor, top_k: int) -> None:
    """Save merged tensors and top heads to files."""
    # Save merged tensors
    torch.save(cie_merged, directory / 'cie_merged.pt')
    torch.save(correct_cie_merged, directory / 'correct_cie_merged.pt')
    
    # Compute and save top heads
    top_heads = compute_top_heads(cie_merged, top_k)
    correct_top_heads = compute_top_heads(correct_cie_merged, top_k)
    
    # Save as JSON
    with open(directory / 'top_heads.json', 'w') as f:
        json.dump(top_heads, f, indent=2)
    
    with open(directory / 'correct_top_heads.json', 'w') as f:
        json.dump(correct_top_heads, f, indent=2)
    
    print(f"Results saved to {directory}")


def cleanup_layer_files(directory: Path, num_layers: int) -> None:
    """Remove individual layer files after successful merging."""
    for layer in range(num_layers):
        cie_path, correct_cie_path = get_layer_file_paths(directory, layer)
        try:
            cie_path.unlink()
            correct_cie_path.unlink()
        except OSError as e:
            print(f"Warning: Failed to remove {cie_path} or {correct_cie_path}: {e}")


def process_directory(directory: Path, num_layers: int, top_k: int) -> None:
    """Process a single directory: merge layers and save results."""
    print(f"Processing {directory}...")
    
    try:
        # Load and merge layers
        cie_merged, correct_cie_merged = load_and_merge_layers(directory, num_layers)
        
        # Save results
        save_results(directory, cie_merged, correct_cie_merged, top_k)
        
        # Clean up individual layer files
        cleanup_layer_files(directory, num_layers)
        
    except Exception as e:
        print(f"Error processing {directory}: {e}")
        return


def main() -> None:
    """Main execution function."""
    args = parse_arguments()
    
    # Find all directories to process
    pattern = f'./{args.output_dir}/*/*/*/*/*'
    directory_paths = [Path(path) for path in glob(pattern)]
    
    if not directory_paths:
        print(f"No directories found matching pattern: {pattern}")
        sys.exit(1)
    
    print(f"Found {len(directory_paths)} directories to check")
    
    # Validate directories
    valid_directories = validate_directories(directory_paths, args.num_layers)
    
    if not valid_directories:
        print("No valid directories found for processing.")
        sys.exit(1)
    
    print(f"Processing {len(valid_directories)} directories...")
    
    # Process each valid directory
    for directory in valid_directories:
        process_directory(directory, args.num_layers, args.top_k)
    
    print("All directories processed successfully!")


if __name__ == '__main__':
    main()