"""
Model evaluation and inference
"""
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import config
from src.utils import AverageMeter, accuracy


class Evaluator:
    """Model evaluation class"""
    
    def __init__(self, model, device, class_names):
        self.model = model
        self.device = device
        self.class_names = class_names
        self.model.to(device)
        self.model.eval()
    
    def evaluate(self, val_loader):
        """Comprehensive evaluation on validation set"""
        all_preds = []
        all_targets = []
        
        top1_acc = AverageMeter()
        top5_acc = AverageMeter()
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                
                # Get predictions
                _, pred = outputs.max(1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                # Calculate accuracy
                acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
                top1_acc.update(acc1[0].item(), inputs.size(0))
                top5_acc.update(acc5[0].item(), inputs.size(0))
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        return {
            'top1_accuracy': top1_acc.avg,
            'top5_accuracy': top5_acc.avg,
            'predictions': all_preds,
            'targets': all_targets,
            'confusion_matrix': confusion_matrix(all_targets, all_preds),
            'classification_report': classification_report(all_targets, all_preds, target_names=self.class_names)
        }
    
    def predict(self, inputs):
        """Single batch prediction"""
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
        
        return predictions.cpu().numpy(), probabilities.cpu().numpy()
    
    def plot_confusion_matrix(self, eval_results, save_path=None):
        """Plot and save confusion matrix"""
        cm = eval_results['confusion_matrix']
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names,
                    cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved: {save_path}")
        
        plt.close()
    
    def plot_training_history(self, train_history, save_path=None):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        axes[0].plot(train_history['train_losses'], label='Train Loss', linewidth=2)
        axes[0].plot(train_history['val_losses'], label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[1].plot(train_history['train_accuracies'], label='Train Accuracy', linewidth=2)
        axes[1].plot(train_history['val_accuracies'], label='Val Accuracy', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history saved: {save_path}")
        
        plt.close()
    
    def print_classification_report(self, eval_results):
        """Print detailed classification report"""
        print("\n" + "="*80)
        print("CLASSIFICATION REPORT")
        print("="*80)
        print(eval_results['classification_report'])
        print("="*80 + "\n")
