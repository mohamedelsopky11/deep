import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

def get_model(num_classes=3, device='cpu'):
    """
    Returns a DeepLabV3+ model with ResNet50 backbone.
    Adjusts the classifier head for num_classes.
    """
    print(f"Loading DeepLabV3+ ResNet50 for {num_classes} classes...")
    try:
        # Use valid weights enum instead of deprecated pretrained=True
        weights = DeepLabV3_ResNet50_Weights.DEFAULT
        model = deeplabv3_resnet50(weights=weights)
    except:
        # Fallback for older torchvision
        model = deeplabv3_resnet50(pretrained=True)
    
    # Replace the auxiliary classifier head
    if model.aux_classifier is not None:
        in_features = model.aux_classifier[4].in_channels
        model.aux_classifier[4] = nn.Conv2d(in_features, num_classes, kernel_size=(1, 1), stride=(1, 1))

    # Replace the main classifier head
    in_features = model.classifier[4].in_channels
    model.classifier[4] = nn.Conv2d(in_features, num_classes, kernel_size=(1, 1), stride=(1, 1))
    
    model.to(device)
    return model

if __name__ == "__main__":
    # Test
    model = get_model()
    x = torch.randn(2, 3, 512, 512)
    y = model(x)['out']
    print(f"Output Shape: {y.shape}")
