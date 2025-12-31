import torch
import torch.nn.functional as F
import numpy as np
import cv2

class FireGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        # grad_output[0] matches output of forward
        self.gradients = grad_output[0]
        
    def __call__(self, x, class_idx=1):
        """
        x: Input tensor (1, 3, H, W)
        class_idx: Target class to visualize (1=Fire)
        """
        self.model.zero_grad()
        output = self.model(x)['out'] # (1, C, H, W)
        
        # Target score: sum of pixels for the class or max pixel? 
        # For segmentation, usually we backprop from the sum of the target class mask
        target = output[:, class_idx, :, :]
        score = target.sum()
        
        score.backward(retain_graph=True)
        
        # Generate CAM
        gradients = self.gradients # (1, K, h, w)
        activations = self.activations # (1, K, h, w)
        
        if gradients is None or activations is None:
            return None
        
        # Global Average Pooling of Gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True) # (1, K, 1, 1)
        
        # Weighted sum of activations
        cam = torch.sum(weights * activations, dim=1, keepdim=True) # (1, 1, h, w)
        
        # ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-7)
        
        # Upsample to input size
        cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        return cam.detach().cpu().numpy()[0, 0] # (H, W)

    def show_cam_on_image(self, img, mask):
        """
        img: RGB (H, W, 3) float 0-1
        mask: (H, W) float 0-1
        """
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + img
        cam = cam / np.max(cam)
        return np.uint8(255 * cam)
