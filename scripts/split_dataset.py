"""
Sequence-Aware Deterministic Dataset Splitting Script.
Ensures no temporal leakage by grouping consecutive video sequence frames together
while balancing environments (City, Forest, Lake) and weather (Foggy, Sunny)
across Train (70%), Validation (15%), and Test (15%) partitions.
"""

import os
import glob
import json
import argparse
import random
from collections import defaultdict

def create_splits(dataset_dir, output_dir, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    
    txt_files = glob.glob(os.path.join(dataset_dir, "*.txt"))
    if not txt_files:
        raise ValueError(f"No annotation files found in {dataset_dir}")
    
    # Group files by sequence key: e.g. augmented_raw_dataset_city_foggy_city_foggy_0_1
    grouped_sequences = defaultdict(list)
    
    for txt_path in txt_files:
        filename = os.path.basename(txt_path)
        base_name = os.path.splitext(filename)[0]
        # corresponding image
        img_png = os.path.join(dataset_dir, f"{base_name}.png")
        img_jpg = os.path.join(dataset_dir, f"{base_name}.jpg")
        
        img_path = img_png if os.path.exists(img_png) else (img_jpg if os.path.exists(img_jpg) else None)
        if not img_path:
            continue
            
        # Parse sequence group
        parts = filename.split("_sequence.")
        if len(parts) > 1:
            seq_group = parts[0]
        else:
            seq_group = filename.split(".")[0]
            
        grouped_sequences[seq_group].append((img_path, txt_path))
        
    print(f"Discovered {len(grouped_sequences)} unique sequence groups with {len(txt_files)} total frames.")
    
    # Stratify by environment and weather category
    stratified_groups = defaultdict(list)
    for seq_key in grouped_sequences.keys():
        if "city" in seq_key:
            env = "city"
        elif "forest" in seq_key:
            env = "forest"
        elif "lake" in seq_key:
            env = "lake"
        else:
            env = "general"
            
        weather = "foggy" if "foggy" in seq_key else ("sunny" if "sunny" in seq_key else "clear")
        strat_key = f"{env}_{weather}"
        stratified_groups[strat_key].append(seq_key)
        
    train_samples = []
    val_samples = []
    test_samples = []
    
    for strat_key, seq_keys in stratified_groups.items():
        random.shuffle(seq_keys)
        n = len(seq_keys)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        
        train_keys = seq_keys[:n_train]
        val_keys = seq_keys[n_train:n_train + n_val]
        test_keys = seq_keys[n_train + n_val:]
        
        if not test_keys and len(val_keys) > 1:
            test_keys = [val_keys.pop()]
            
        for k in train_keys:
            train_samples.extend(grouped_sequences[k])
        for k in val_keys:
            val_samples.extend(grouped_sequences[k])
        for k in test_keys:
            test_samples.extend(grouped_sequences[k])
            
    print(f"\nSplit Distribution:")
    print(f"  Train : {len(train_samples)} frames ({len(train_samples)/len(txt_files)*100:.1f}%)")
    print(f"  Val   : {len(val_samples)} frames ({len(val_samples)/len(txt_files)*100:.1f}%)")
    print(f"  Test  : {len(test_samples)} frames ({len(test_samples)/len(txt_files)*100:.1f}%)")
    print(f"  Total : {len(train_samples) + len(val_samples) + len(test_samples)} frames")
    
    # Save split manifests using normalized project-root relative paths
    def save_split_file(samples, split_name):
        txt_out = os.path.join(output_dir, f"{split_name}.txt")
        json_out = os.path.join(output_dir, f"{split_name}.json")
        
        with open(txt_out, "w") as f:
            for img_p, txt_p in samples:
                rel_img = os.path.relpath(img_p, os.getcwd()).replace("\\", "/")
                rel_txt = os.path.relpath(txt_p, os.getcwd()).replace("\\", "/")
                f.write(f"{rel_img},{rel_txt}\n")
                
        json_records = [{"image": os.path.abspath(img_p), "label": os.path.abspath(txt_p)} for img_p, txt_p in samples]
        with open(json_out, "w") as f:
            json.dump(json_records, f, indent=2)
            
    save_split_file(train_samples, "train")
    save_split_file(val_samples, "val")
    save_split_file(test_samples, "test")
    
    print(f"\nManifest files successfully written to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split drone dataset into sequence-aware train/val/test splits.")
    parser.add_argument("--dataset-dir", type=str, default="curated_datasets/obj_det_base", help="Path to raw dataset directory")
    parser.add_argument("--output-dir", type=str, default="data/splits", help="Path to output split manifests")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    create_splits(args.dataset_dir, args.output_dir, seed=args.seed)
