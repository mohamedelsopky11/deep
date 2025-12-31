import numpy as np
import cv2

class SeverityEstimator:
    def __init__(self):
        self.levels = {
            "Low": (0, 20),
            "Medium": (20, 50),
            "High": (50, 80),
            "Critical": (80, 100)
        }
        
    def calculate_score(self, image, mask, prob_map=None):
        """
        image: Original RGB image (numpy)
        mask: Segmentation mask (0,1,2)
        prob_map: Probability map from model (optional)
        """
        h, w = mask.shape
        total_pixels = h * w
        
        # 1. Area Ratios
        fire_pixels = np.count_nonzero(mask == 1)
        smoke_pixels = np.count_nonzero(mask == 2)
        
        fire_ratio = fire_pixels / total_pixels
        smoke_ratio = smoke_pixels / total_pixels
        
        # 2. Intensity (Fire ROI)
        intensity_score = 0
        if fire_pixels > 0:
            # Extract Fire ROI
            fire_roi = image[mask == 1]
            # Simple heuristic: Mean brightness (V in HSV) or R channel
            # Use Red channel
            mean_red = np.mean(fire_roi[:, 0]) # R is index 0 if RGB, 2 if BGR. Assumes RGB here.
            intensity_score = mean_red / 255.0
            
        # 3. Formula
        # Weights: Fire area is most critical. Intensity matters. Smoke adds context.
        # Score = (FireArea * 400 clipped to 60) + (SmokeArea * 50) + (Intensity * 20)
        # This is a heuristic.
        
        # If fire covers 10% of screen, that's already HUGE. 
        # So 0.1 ratio -> 40 points?
        base_score = (fire_ratio * 100 * 5) + (smoke_ratio * 100 * 1)
        
        # Add intensity factor (make it punchier if bright)
        final_score = base_score + (intensity_score * 10)
        
        # Clip
        final_score = min(100, max(0, final_score))
        
        label = self.get_label(final_score)
        
        return {
            "score": round(final_score, 2),
            "label": label,
            "details": {
                "fire_ratio": round(fire_ratio, 4),
                "smoke_ratio": round(smoke_ratio, 4),
                "intensity": round(intensity_score, 2)
            }
        }
        
    def get_label(self, score):
        for label, (low, high) in self.levels.items():
            if low <= score <= high:
                return label
        return "Critical" # Fallback
