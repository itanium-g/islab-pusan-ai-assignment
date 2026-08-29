"""
Dataset Packaging Utility.
Creates a portable, optimized zip archive of the dataset (or cached dataset)
for instant upload to Kaggle Datasets, Google Drive, or Hugging Face.
"""

import os
import zipfile
import argparse
from tqdm import tqdm

def package_dataset(source_dir, output_zip, exclude_raw=False):
    os.makedirs(os.path.dirname(os.path.abspath(output_zip)), exist_ok=True)
    
    file_list = []
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith((".jpg", ".png", ".txt", ".json", ".yaml")):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, source_dir)
                file_list.append((file_path, rel_path))
                
    print(f"Packaging {len(file_list)} files from {source_dir} -> {output_zip}...")
    
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path, rel_path in tqdm(file_list, desc="Archiving"):
            zipf.write(file_path, rel_path)
            
    size_mb = os.path.getsize(output_zip) / (1024 * 1024)
    print(f"\nPackage created successfully: {output_zip} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package dataset into a zip archive.")
    parser.add_argument("--source-dir", type=str, default="data/cached_640", help="Directory to package")
    parser.add_argument("--output-zip", type=str, default="data/drone_dataset_640.zip", help="Output zip path")
    args = parser.parse_args()
    
    package_dataset(args.source_dir, args.output_zip)
