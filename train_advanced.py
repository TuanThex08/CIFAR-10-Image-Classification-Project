"""
Advanced Training Example with Custom Features
This script demonstrates advanced training techniques including:
- Learning rate warm-up
- Gradient accumulation
- Mixed precision training
- Advanced logging
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import time
from pathlib import Path
from src.config import config
from src.model import resnet18
from src.dataset import get_data_loaders
from src.utils import set_seed, get_device, accuracy, AverageMeter


def train_with_amp(model, train_loader, val_loader, num_epochs=200):
    """
    Training with Automatic Mixed Precision (AMP)
    Reduces memory usage and speeds up training on modern GPUs
    """
    
    device = get_device()
    model.to(device)
    
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.LEARNING_RATE,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY,
        nesterov=True
    )
    
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    
    print("\n" + "="*60)
    print("Training with Automatic Mixed Precision (AMP)")
    print("="*60)
    
    best_acc = 0.0
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = AverageMeter()
        train_acc = AverageMeter()
        
        start_time = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Mixed precision training
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # Metrics
            acc1, _ = accuracy(outputs, targets, topk=(1, 5))
            train_loss.update(loss.item(), inputs.size(0))
            train_acc.update(acc1[0].item(), inputs.size(0))
            
            if (batch_idx + 1) % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch} [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {train_loss.avg:.4f} | Acc: {train_acc.avg:.2f}% "
                      f"| Time: {elapsed:.1f}s")
        
        # Validation
        model.eval()
        val_loss = AverageMeter()
        val_acc = AverageMeter()
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                acc1, _ = accuracy(outputs, targets, topk=(1, 5))
                val_loss.update(loss.item(), inputs.size(0))
                val_acc.update(acc1[0].item(), inputs.size(0))
        
        print(f"Val Loss: {val_loss.avg:.4f} | Val Acc: {val_acc.avg:.2f}%\n")
        
        if val_acc.avg > best_acc:
            best_acc = val_acc.avg
            torch.save(model.state_dict(), config.MODEL_DIR / "best_amp_model.pt")
    
    print(f"Best validation accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    set_seed(config.SEED)
    
    # Load data
    print("Loading CIFAR-10...")
    train_loader, val_loader, _ = get_data_loaders()
    
    # Create model
    model = resnet18(num_classes=config.NUM_CLASSES)
    
    # Train with AMP
    train_with_amp(model, train_loader, val_loader, num_epochs=config.NUM_EPOCHS)
