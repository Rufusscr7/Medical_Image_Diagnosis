import torch
import torch.nn as nn
from torchvision import models


def create_model(num_classes=7, pretrained=True):
    """
    Builds a ResNet-18 model adapted for skin lesion classification.
    Replaces the default ImageNet 1000-class head with a linear layer for `num_classes`.
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    # Replace the final fully connected classification layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


if __name__ == "__main__":
    model = create_model(num_classes=7)
    print(f"ResNet-18 initialized successfully with {model.fc.out_features} output classes.")

    # Quick test forward pass with dummy tensor
    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Sample forward pass output shape: {output.shape}")