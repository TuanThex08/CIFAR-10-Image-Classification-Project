"""
Training module for CNN model
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, LinearLR
import time
from pathlib import Path
from src.config import config
from src.utils import AverageMeter, accuracy, save_checkpoint, save_best_model


class Trainer:
    """Main training class"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
        # Move model to device
        self.model.to(device)
        
        # Optimizer
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=config.LEARNING_RATE,
            momentum=config.MOMENTUM,
            weight_decay=config.WEIGHT_DECAY,
            nesterov=True
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Scheduler
        self.scheduler = self._get_scheduler()
        
        # Metrics
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        self.best_acc = 0.0
        self.patience_counter = 0
    
    def _get_scheduler(self):
        """Get learning rate scheduler"""
        if config.SCHEDULER_TYPE == "cosine":
            return CosineAnnealingLR(self.optimizer, T_max=config.NUM_EPOCHS)
        elif config.SCHEDULER_TYPE == "step":
            return StepLR(self.optimizer, step_size=config.STEP_SIZE, gamma=config.GAMMA)
        elif config.SCHEDULER_TYPE == "linear":
            return LinearLR(self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=config.NUM_EPOCHS)
        else:
            return StepLR(self.optimizer, step_size=config.STEP_SIZE, gamma=config.GAMMA)
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        
        losses = AverageMeter()
        top1_acc = AverageMeter()
        top5_acc = AverageMeter()
        
        start_time = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Calculate accuracy
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            
            losses.update(loss.item(), inputs.size(0))
            top1_acc.update(acc1[0].item(), inputs.size(0))
            top5_acc.update(acc5[0].item(), inputs.size(0))
            
            if (batch_idx + 1) % config.LOG_INTERVAL == 0:
                elapsed_time = time.time() - start_time
                print(f"[Epoch {epoch}] [{batch_idx + 1}/{len(train_loader)}] "
                      f"Loss: {losses.avg:.4f} | Top-1: {top1_acc.avg:.2f}% | "
                      f"Top-5: {top5_acc.avg:.2f}% | Time: {elapsed_time:.1f}s")
        
        self.train_losses.append(losses.avg)
        self.train_accuracies.append(top1_acc.avg)
        
        return losses.avg, top1_acc.avg
    
    def validate(self, val_loader, epoch):
        """Validate the model"""
        self.model.eval()
        
        losses = AverageMeter()
        top1_acc = AverageMeter()
        top5_acc = AverageMeter()
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
                
                losses.update(loss.item(), inputs.size(0))
                top1_acc.update(acc1[0].item(), inputs.size(0))
                top5_acc.update(acc5[0].item(), inputs.size(0))
        
        self.val_losses.append(losses.avg)
        self.val_accuracies.append(top1_acc.avg)
        
        print(f"[Validation] Loss: {losses.avg:.4f} | Top-1: {top1_acc.avg:.2f}% | Top-5: {top5_acc.avg:.2f}%")
        
        return losses.avg, top1_acc.avg
    
    def train(self, train_loader, val_loader):
        """Full training loop"""
        print(f"Starting training for {config.NUM_EPOCHS} epochs...")
        print(f"Number of trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        for epoch in range(1, config.NUM_EPOCHS + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{config.NUM_EPOCHS}")
            print(f"{'='*60}")
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_loss, val_acc = self.validate(val_loader, epoch)
            
            # Update scheduler
            self.scheduler.step()
            print(f"Learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint
            if epoch % config.CHECKPOINT_INTERVAL == 0:
                checkpoint_metrics = {
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc
                }
                save_checkpoint(self.model, self.optimizer, epoch, checkpoint_metrics)
            
            # Save best model
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.patience_counter = 0
                best_metrics = {'accuracy': val_acc, 'loss': val_loss}
                save_best_model(self.model, best_metrics)
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
        
        print("\n" + "="*60)
        print(f"Training completed! Best accuracy: {self.best_acc:.4f}")
        print("="*60)
        
        return {
            'best_accuracy': self.best_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
