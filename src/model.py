import torch
import torch.nn as nn
from torchvision import models


def create_model(num_classes=7):

    # Load pretrained ResNet18
    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT
    )

    # Get number of input features
    num_features = model.fc.in_features

    # Replace final layer for 7 skin disease classes
    model.fc = nn.Linear(
        num_features,
        num_classes
    )

    return model


# Test the model
if __name__ == "__main__":

    model = create_model()

    print("=" * 50)
    print("RESNET18 MODEL CREATED SUCCESSFULLY")
    print("=" * 50)

    print("\nNumber of output classes:")
    print(model.fc.out_features)

    print("\nModel:")
    print(model)