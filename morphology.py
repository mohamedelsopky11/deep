import cv2
import numpy as np

class MorphologyProcessor:
    def __init__(self, kernel_size=3):
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
            
    def process_mask(self, mask):
        """
        Apply morphological operations to clean up the mask.
        mask: np.array (H, W) with class indices [0, 1, 2]
        """
        # Separate channels for independent processing
        fire_mask = (mask == 1).astype(np.uint8) * 255
        smoke_mask = (mask == 2).astype(np.uint8) * 255
        
        fire_clean = self._clean_channel(fire_mask)
        smoke_clean = self._clean_channel(smoke_mask)
        
        # Merge back: Fire takes precedence if overlaps (for safety)
        final_mask = np.zeros_like(mask)
        final_mask[smoke_clean > 0] = 2
        final_mask[fire_clean > 0] = 1
        
        return final_mask

    def _clean_channel(self, binary_mask):
        """
        Opening (remove noise) -> Closing (fill gaps) -> Fill Holes
        """
        # Opening: erode then dilate -> removes small noise
        opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, self.kernel)
        
        # Closing: dilate then erode -> fills small holes
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self.kernel)
        
        # Fill Holes (Contour based)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(closed)
        for cnt in contours:
            # Filter small objects by area
            if cv2.contourArea(cnt) > 50: # min area
                cv2.drawContours(filled, [cnt], -1, 255, thickness=cv2.FILLED)
                
        return filled

if __name__ == "__main__":
    # Test
    mp = MorphologyProcessor()
    fake_mask = np.zeros((100, 100), dtype=np.uint8)
    fake_mask[20:30, 20:30] = 1 # box
    fake_mask[25:35, 25:35] = 0 # hole
    fake_mask[80:82, 80:82] = 1 # noise
    
    res = mp._clean_channel(fake_mask * 255)
    print(f"Original items: {np.count_nonzero(fake_mask)}")
    print(f"Result items: {np.count_nonzero(res)}")
