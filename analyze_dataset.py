import os
import cv2
import glob
import numpy as np
from collections import Counter

DATASET_ROOT = r"d:/Programming/Python/CV Project/FireData"

def scan_folder(folder_path):
    print(f"\nScanning: {folder_path}")
    if not os.path.exists(folder_path):
        print(f"[X] Folder not found: {folder_path}")
        return

    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, '**', ext), recursive=True))
    
    print(f"Found {len(image_files)} images.")
    
    if len(image_files) == 0:
        return

    # Analyze a sample
    resolutions = []
    corrupted = 0
    
    # Check first 1000 images or 10% for speed if massive
    sample_size = min(len(image_files), 1000)
    print(f"Analyzing sample of {sample_size} images...")
    
    for img_path in image_files[:sample_size]:
        try:
            img = cv2.imread(img_path)
            if img is None:
                corrupted += 1
                continue
            resolutions.append(img.shape[:2]) # H, W
        except Exception:
            corrupted += 1
            
    if resolutions:
        unique_res = Counter(resolutions)
        print(f"Top 5 Resolutions: {unique_res.most_common(5)}")
        min_res = min(resolutions, key=lambda x: x[0]*x[1])
        max_res = max(resolutions, key=lambda x: x[0]*x[1])
        print(f"Min Resolution: {min_res}")
        print(f"Max Resolution: {max_res}")
    
    print(f"Corrupted images in sample: {corrupted}")
    return image_files


def main():
    report_path = "dataset_report_v2.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        def log(msg):
            print(msg)
            f.write(msg + "\n")
            
        log("--- FireData Analysis ---")
        
        # 1. Fire Dataset
        fire_dataset_path = os.path.join(DATASET_ROOT, "fire_dataset")
        log(f"\nScanning: {fire_dataset_path}")
        if not os.path.exists(fire_dataset_path):
             log(f"[X] Folder not found: {fire_dataset_path}")
        else:
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif']
            image_files = []
            for ext in image_extensions:
                image_files.extend(glob.glob(os.path.join(fire_dataset_path, '**', ext), recursive=True))
            
            log(f"Found {len(image_files)} images.")
            
            if len(image_files) > 0:
                resolutions = []
                corrupted = 0
                sample_size = min(len(image_files), 1000)
                log(f"Analyzing sample of {sample_size} images...")
                
                for img_path in image_files[:sample_size]:
                    try:
                        img = cv2.imread(img_path)
                        if img is None:
                            corrupted += 1
                            continue
                        resolutions.append(img.shape[:2]) # H, W
                    except Exception:
                        corrupted += 1
                        
                if resolutions:
                    unique_res = Counter(resolutions)
                    log(f"Top 5 Resolutions: {unique_res.most_common(5)}")
                    min_res = min(resolutions, key=lambda x: x[0]*x[1])
                    max_res = max(resolutions, key=lambda x: x[0]*x[1])
                    log(f"Min Resolution: {min_res}")
                    log(f"Max Resolution: {max_res}")
                log(f"Corrupted images in sample: {corrupted}")

        # 2. Acoustic Dataset
        acoustic_path = os.path.join(DATASET_ROOT, "acoustic_dataset")
        log(f"\nScanning: {acoustic_path}")
        if not os.path.exists(acoustic_path):
             log(f"[X] Folder not found: {acoustic_path}")
        else:
             # Just simplified repetition for time
            image_files_ac = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                image_files_ac.extend(glob.glob(os.path.join(acoustic_path, '**', ext), recursive=True))
            log(f"Found {len(image_files_ac)} images.")

        # 3. Fire and Smoke Dataset
        fs_dataset_path = os.path.join(DATASET_ROOT, "Fire and Smoke Dataset")
        log(f"\nScanning: {fs_dataset_path}")
        if not os.path.exists(fs_dataset_path):
             log(f"[X] Folder not found: {fs_dataset_path}")
        else:
            image_files_fs = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                image_files_fs.extend(glob.glob(os.path.join(fs_dataset_path, '**', ext), recursive=True))
            log(f"Found {len(image_files_fs)} images.")


if __name__ == "__main__":
    main()
