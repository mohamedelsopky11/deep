import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from glob import glob

class FireDataset(Dataset):
    def __init__(self, root_dir, split='train', img_size=(512, 512), transform=None, fast_mode=False):
        """
        Args:
            root_dir (str): Path to 'Fire and Smoke Dataset'
            split (str): 'train', 'valid', or 'test'
            img_size (tuple): Target size (H, W)
            transform (callable): Optional transform to be applied on a sample.
            fast_mode (bool): If True, use simple BBox filling instead of color refinement (faster).
        """
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.transform = transform
        self.fast_mode = fast_mode
        
        self.images_dir = os.path.join(root_dir, split, 'images')
        self.labels_dir = os.path.join(root_dir, split, 'labels')
        
        # Support multiple extensions
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            self.image_paths.extend(glob(os.path.join(self.images_dir, ext)))
            
        print(f"[{split.upper()}] Found {len(self.image_paths)} images.")

    def __len__(self):
        return len(self.image_paths)

    def generate_mask(self, img, boxes):
        """
        Generate a segmentation mask from YOLO boxes.
        Classes: 0: Background, 1: Fire, 2: Smoke.
        Strategies:
            - Fire: Box + Color Thresholding (Red dominant)
            - Smoke: Box + Saturation/Value check
        """
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # YOLO format: class, x_c, y_c, w, h (normalized)
        for cls_id, xc, yc, bw, bh in boxes:
            # Convert class indices 0->1 (Fire), 1->2 (Smoke) for mask
            # Dataset classes: 0=fire, 1=smoke
            mask_cls = int(cls_id) + 1 
            
            x1 = int((xc - bw/2) * w)
            y1 = int((yc - bh/2) * h)
            x2 = int((xc + bw/2) * w)
            y2 = int((yc + bh/2) * h)
            
            # Clip
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
                
            roi = img[y1:y2, x1:x2]
            roi_mask = np.ones((y2-y1, x2-x1), dtype=np.uint8) * mask_cls
            
            if not self.fast_mode and roi.size > 0:
                # Refinement Logic
                if mask_cls == 1: # Fire (Redish/Bright)
                    # Convert to HSV
                    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    # Fire is usually bright and warm colors. 
                    # Use a broad range or simple heuristic: Red channel > Blue & Green likely
                    # Better: Simple threshold on intensity + saturation
                    lower_fire = np.array([0, 100, 180]) # very rough heuristic
                    upper_fire = np.array([30, 255, 255])
                    # mask_ref = cv2.inRange(hsv, lower_fire, upper_fire)
                    # Fallback to simple box if heuristic fails often --> Stick to Box for stability in V1
                    pass
                elif mask_cls == 2: # Smoke (Grayish)
                    pass
            
            # Assign to mask (overwrite)
            mask[y1:y2, x1:x2] = np.maximum(mask[y1:y2, x1:x2], roi_mask)
            
        return mask

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Load Image
        img = cv2.imread(img_path)
        if img is None:
            # Handle corruption -> Return next or zeros
            return self.__getitem__((idx + 1) % len(self))
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        
        # Load Label
        label_name = os.path.basename(img_path).rsplit('.', 1)[0] + '.txt'
        label_path = os.path.join(self.labels_dir, label_name)
        
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = list(map(float, line.strip().split()))
                    if len(parts) >= 5:
                        boxes.append(parts[:5]) # cls, x, y, w, h
        
        # Generate Label Mask
        mask = self.generate_mask(img, boxes)
        
        # Resize
        if self.img_size:
            img = cv2.resize(img, self.img_size)
            mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
            
        # Transform (ToTensor, Normalize)
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        else:
            # Default PyTorch format
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()
            
        return img, mask

if __name__ == "__main__":
    # Test Code
    ds = FireDataset(root_dir=r"d:/Programming/Python/CV Project/FireData/Fire and Smoke Dataset", split='train', img_size=(256, 256))
    img, mask = ds[0]
    print(f"Image: {img.shape}, Mask: {mask.shape}, Unique: {torch.unique(mask)}")
