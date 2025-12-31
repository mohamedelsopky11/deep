import torch
import cv2
import numpy as np
import os
import time
from src.model import get_model
from src.morphology import MorphologyProcessor
from src.severity import SeverityEstimator
from src.gradcam import FireGradCAM

class FireRiskSystem:
    def __init__(self, model_path="models/fire_seg_best.pth", device='cpu'):
        self.device = device
        self.model = get_model(num_classes=3, device=device)
        
        if os.path.exists(model_path):
            state = torch.load(model_path, map_location=device)
            self.model.load_state_dict(state)
            print("Model loaded.")
        else:
            print(f"Warning: Model not found at {model_path}. Using random weights.")
            
        self.model.eval()
        
        # Modules
        self.morph = MorphologyProcessor()
        self.severity = SeverityEstimator()
        
        # Explainability (Targeting the last block of ResNet backbone)
        # DeepLab in torchvision: model.backbone.layer4
        self.grad_cam = FireGradCAM(self.model, self.model.backbone.layer4)

    def run_inference(self, image_path, save_dir="reports/inference_results"):
        t0 = time.time()
        
        # 1. Load Image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Image not found")
            
        orig_h, orig_w = image.shape[:2]
        
        # Preprocess
        # Resize to fixed size for model (256x256 or 512x512)
        input_size = (256, 256) 
        img_resized = cv2.resize(image, input_size)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)
        
        # 2. Inference
        with torch.no_grad():
            output = self.model(tensor)['out']
            # Softmax
            probs = torch.softmax(output, dim=1)
            pred_mask = torch.argmax(probs, dim=1).cpu().numpy()[0] # (H, W)
            
        # 3. Morphology
        clean_mask = self.morph.process_mask(pred_mask)
        # Resize mask back to original
        clean_mask_full = cv2.resize(clean_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        
        # 4. Severity
        # Use simple mask for now, assume Fire=1, Smoke=2
        sev_res = self.severity.calculate_score(image, clean_mask_full)
        
        # 5. Grad-CAM
        # Get heatmap for Fire (Index 1)
        cam = self.grad_cam(tensor, class_idx=1)
        if cam is not None:
             # Resize CAM to original
             cam = cv2.resize(cam, (orig_w, orig_h))
             cam_viz = self.grad_cam.show_cam_on_image(image/255.0, cam)
        else:
             cam_viz = np.zeros_like(image)

        # 6. Visualization (Overlay)
        # Create user-friendly overlay
        overlay = image.copy()
        # Fire = Red, Smoke = Gray
        overlay[clean_mask_full == 1] = [0, 0, 255] # Red BGR
        overlay[clean_mask_full == 2] = [128, 128, 128] # Gray
        
        alpha = 0.5
        blended = cv2.addWeighted(overlay, alpha, image, 1-alpha, 0)
        
        # 7. Save
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            timestamp = int(time.time())
            base_name = os.path.basename(image_path).rsplit('.', 1)[0]
            
            cv2.imwrite(f"{save_dir}/{base_name}_{timestamp}_overlay.jpg", blended)
            cv2.imwrite(f"{save_dir}/{base_name}_{timestamp}_cam.jpg", cam_viz)
            
        return {
            "processed_image": blended,
            "mask": clean_mask_full,
            "grad_cam": cam_viz,
            "severity": sev_res,
            "latency": time.time() - t0
        }

if __name__ == "__main__":
    # Test on a real image
    import glob
    sys = FireRiskSystem()
    files = glob.glob(r"d:/Programming/Python/CV Project/FireData/fire_dataset/fire_images/*.jpg")
    if files:
        res = sys.run_inference(files[0])
        print("Result:", res['severity'])
