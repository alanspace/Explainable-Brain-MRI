import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, resnet50, ResNet50_Weights

def get_model(model_name='efficientnet_b0', num_classes=4, pretrained=True, freeze_feature_extractor=False):
    """
    Returns a customized model for Brain Tumor Classification.
    Supported: 'efficientnet_b0', 'resnet50'
    """
    if model_name == 'efficientnet_b0':
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        classifier_layer = model.classifier
    elif model_name == 'resnet50':
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        classifier_layer = model.fc
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    # Freeze feature extractor layers if requested
    if pretrained and freeze_feature_extractor:
        for name, param in model.named_parameters():
            if classifier_layer not in name:
                param.requires_grad = False
            
    return model
