"""
Inference Example - How to use the trained model for predictions
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from src.config import config
from src.model import resnet18
from src.utils import get_device, load_model


def predict_image(image_path, model, device, class_names):
    """
    Predict class for a single image
    
    Args:
        image_path: Path to image
        model: Trained model
        device: GPU or CPU
        class_names: List of class names
    
    Returns:
        predicted_class, confidence, all_probabilities
    """
    
    # Preprocess image
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        ),
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    # Inference
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return class_names[predicted_class], confidence, probabilities[0].cpu().numpy()


def predict_batch(image_tensor, model, device, class_names):
    """
    Predict classes for a batch of images
    
    Args:
        image_tensor: Batch tensor of shape (B, C, H, W)
        model: Trained model
        device: GPU or CPU
        class_names: List of class names
    
    Returns:
        predictions, confidences, probabilities
    """
    
    image_tensor = image_tensor.to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        predicted_classes = torch.argmax(probabilities, dim=1)
        confidences = probabilities.gather(1, predicted_classes.unsqueeze(1)).squeeze()
    
    predictions = [class_names[idx] for idx in predicted_classes.cpu().numpy()]
    
    return predictions, confidences.cpu().numpy(), probabilities.cpu().numpy()


def visualize_predictions(image, predictions, class_names, top_k=3):
    """
    Visualize predictions with confidence scores
    """
    
    top_k_indices = np.argsort(predictions)[::-1][:top_k]
    
    print("Top-3 Predictions:")
    for i, idx in enumerate(top_k_indices):
        print(f"  {i+1}. {class_names[idx]}: {predictions[idx]:.4f}")


# Example usage
if __name__ == "__main__":
    device = get_device()
    
    # Load model
    print("Loading trained model...")
    model = resnet18(num_classes=config.NUM_CLASSES)
    if not load_model(model, "best_model.pt"):
        print("Model not found. Please train first.")
        exit(1)
    
    model.to(device)
    
    # CIFAR-10 classes
    class_names = ('plane', 'car', 'bird', 'cat', 'deer', 
                  'dog', 'frog', 'horse', 'ship', 'truck')
    
    print("\nModel is ready for inference!")
    print(f"Use predict_image() or predict_batch() for predictions")
