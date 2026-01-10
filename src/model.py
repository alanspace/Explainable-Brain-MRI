import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def get_model(num_classes=4, pretrained=True, freeze_feature_extractor=True):
    """
    Returns a ResNet18 model customized for Brain Tumor Classification.
    """
    if pretrained:
        weights = ResNet18_Weights.DEFAULT
    else:
        weights = None
        
    model = resnet18(weights=weights)
    
    # Freeze feature extractor layers if requested
    if pretrained and freeze_feature_extractor:
        for param in model.parameters():
            param.requires_grad = False
            
    # Replace the final fully connected layer
    # ResNet18 fc input features = 512
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model
