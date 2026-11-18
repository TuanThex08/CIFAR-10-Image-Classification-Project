"""
Utility functions for training and evaluation
"""
import torch
import random
import numpy as np
from pathlib import Path
from src.config import config


def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Return the configured device, falling back to CPU when CUDA is unavailable."""
    requested_device = str(config.DEVICE).lower()

    if requested_device.startswith("cuda"):
        if torch.cuda.is_available():
            device = torch.device(config.DEVICE)
            print(f"Using GPU: {torch.cuda.get_device_name(device.index or 0)}")
            return device

        # A CPU-only PyTorch build or a missing/incompatible NVIDIA driver
        # should not prevent inference in the Streamlit app.
        print("CUDA is unavailable; falling back to CPU.")

    device = torch.device("cpu")
    print("Using CPU")
    return device


def save_checkpoint(model, optimizer, epoch, metrics, filename="checkpoint.pt"):
    """Save model checkpoint"""
    checkpoint_path = config.MODEL_DIR / filename
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }, checkpoint_path)
    
    print(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(model, optimizer, filename="checkpoint.pt"):
    """Load model checkpoint"""
    checkpoint_path = config.MODEL_DIR / filename
    
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return 0
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded: {checkpoint_path}")
    return checkpoint['epoch']


def save_best_model(model, metrics, filename="best_model.pt"):
    """Save best model based on accuracy"""
    best_path = config.MODEL_DIR / filename
    
    torch.save(model.state_dict(), best_path)
    
    print(f"Best model saved: {best_path}")
    print(f"Accuracy: {metrics['accuracy']:.4f}, Loss: {metrics['loss']:.4f}")


def load_model(model, filename="best_model.pt"):
    """Load trained model"""
    model_path = config.MODEL_DIR / filename
    
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return False
    
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    print(f"Model loaded: {model_path}")
    return True


def count_parameters(model):
    """Count total trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter:
    """Track average metrics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Calculate top-k accuracy on CPU to avoid fragile CUDA top-k kernels."""
    with torch.no_grad():
        # CIFAR-10 outputs are tiny (batch x 10), so this transfer is cheap
        # and keeps metric calculation independent from CUDA kernel state.
        output = output.detach().float().cpu()
        target = target.detach().cpu()
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
