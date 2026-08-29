"""
High-Speed Parallel Dataset Preprocessor & Caching Utility.
Resizes 2560x1440 uncompressed images into high-quality letterboxed 640x640 images
and updates bounding box coordinate annotations.
Reduces dataset footprint from 9.1 GB to ~150 MB, speeding up epoch training time by 10x-30x.
"""

import os
import glob
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from PIL import Image
import numpy as np
from tqdm import tqdm

def letterbox_image(image, target_size=(640, 640)):
    """
    Resize image with padding to target_size while preserving aspect ratio.
    Returns: padded_image, (scale, pad_x, pad_y, original_w, original_h)
    """
    orig_w, orig_h = image.size
    target_w, target_h = target_size
    
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    
    resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    
    # Create blank canvas with neutral grey padding (114)
    padded = Image.new("RGB", (target_w, target_h), (114, 114, 114))
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    padded.paste(resized, (pad_x, pad_y))
    
    return padded, (scale, pad_x, pad_y, orig_w, orig_h, new_w, new_h)

def process_single_sample(args):
    img_path, txt_path, out_img_dir, out_lbl_dir, target_size, quality = args
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    
    out_img_path = os.path.join(out_img_dir, f"{base_name}.jpg")
    out_txt_path = os.path.join(out_lbl_dir, f"{base_name}.txt")
    
    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            padded_img, meta = letterbox_image(img, target_size)
            padded_img.save(out_img_path, "JPEG", quality=quality, optimize=True)
            
        scale, pad_x, pad_y, orig_w, orig_h, new_w, new_h = meta
        target_w, target_h = target_size
        
        new_annotations = []
        if os.path.exists(txt_path):
            with open(txt_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cls_id = parts[0]
                    # Normalized center x, y, width, height in original image
                    x_c_orig = float(parts[1]) * orig_w
                    y_c_orig = float(parts[2]) * orig_h
                    w_orig = float(parts[3]) * orig_w
                    h_orig = float(parts[4]) * orig_h
                    
                    # Transform to letterboxed coordinates
                    x_c_new = x_c_orig * scale + pad_x
                    y_c_new = y_c_orig * scale + pad_y
                    w_new = w_orig * scale
                    h_new = h_orig * scale
                    
                    # Normalize relative to target size (640x640)
                    norm_xc = np.clip(x_c_new / target_w, 0.0, 1.0)
                    norm_yc = np.clip(y_c_new / target_h, 0.0, 1.0)
                    norm_w = np.clip(w_new / target_w, 0.0, 1.0)
                    norm_h = np.clip(h_new / target_h, 0.0, 1.0)
                    
                    new_annotations.append(f"{cls_id} {norm_xc:.6f} {norm_yc:.6f} {norm_w:.6f} {norm_h:.6f}")
                    
        with open(out_txt_path, "w") as f:
            for ann in new_annotations:
                f.write(ann + "\n")
                
        return True
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return False

def preprocess_dataset(src_dir, dest_dir, target_size=(640, 640), quality=95, num_workers=8):
    out_img_dir = os.path.join(dest_dir, "images")
    out_lbl_dir = os.path.join(dest_dir, "labels")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)
    
    txt_files = glob.glob(os.path.join(src_dir, "*.txt"))
    samples = []
    
    for txt_path in txt_files:
        base_name = os.path.splitext(os.path.basename(txt_path))[0]
        img_png = os.path.join(src_dir, f"{base_name}.png")
        img_jpg = os.path.join(src_dir, f"{base_name}.jpg")
        img_path = img_png if os.path.exists(img_png) else (img_jpg if os.path.exists(img_jpg) else None)
        
        if img_path:
            samples.append((img_path, txt_path, out_img_dir, out_lbl_dir, target_size, quality))
            
    print(f"Starting parallel preprocessing for {len(samples)} samples to size {target_size}...")
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(process_single_sample, samples), total=len(samples), desc="Preprocessing", mininterval=2.0, ncols=80))
        
    success = sum(results)
    print(f"\nPreprocessing Complete: {success}/{len(samples)} images converted and cached to {dest_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess and cache high-resolution drone dataset.")
    parser.add_argument("--src-dir", type=str, default="curated_datasets/obj_det_base", help="Source directory containing raw images and txts")
    parser.add_argument("--dest-dir", type=str, default="data/cached_640", help="Destination directory for cached preprocessed dataset")
    parser.add_argument("--img-size", type=int, default=640, help="Target square image dimension (e.g. 640)")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality level (1-100)")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel worker threads")
    args = parser.parse_args()
    
    preprocess_dataset(args.src_dir, args.dest_dir, target_size=(args.img_size, args.img_size), quality=args.quality, num_workers=args.workers)
