"""
Computer Vision CNN - CIFAR-10 Classification
Main entry point for training and evaluation

This project implements a ResNet-18 architecture for CIFAR-10 image classification
with modern training techniques including data augmentation, learning rate scheduling,
and early stopping.

"""

import torch
import argparse
from pathlib import Path
from src.config import config, Config
from src.model import resnet18, SimpleCNN
from src.dataset import get_data_loaders
from src.trainer import Trainer
from src.evaluator import Evaluator
from src.utils import set_seed, get_device, load_model, count_parameters


def train(args):
    """Training pipeline"""

    # Apply command-line training overrides before creating the data/model pipeline.
    config.BATCH_SIZE = args.batch_size
    config.NUM_EPOCHS = args.epochs
    config.LEARNING_RATE = args.lr
    
    # Set seeds for reproducibility
    set_seed(config.SEED)
    
    # Get device
    device = get_device()
    
    # Get data loaders
    print("\n[*] Loading CIFAR-10 dataset...")
    train_loader, val_loader, class_names = get_data_loaders(batch_size=config.BATCH_SIZE)
    print(f"[✓] Dataset loaded successfully")
    print(f"    Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create model
    print("\n[*] Creating model...")
    if args.model == "resnet18":
        model = resnet18(num_classes=config.NUM_CLASSES)
    else:
        model = SimpleCNN(num_classes=config.NUM_CLASSES)
    
    num_params = count_parameters(model)
    print(f"[✓] Model created: {args.model}")
    print(f"    Total parameters: {num_params:,}")
    
    # Create trainer
    trainer = Trainer(model, device)
    
    # Train model
    print("\n[*] Starting training...")
    history = trainer.train(train_loader, val_loader)
    
    # Evaluate best model
    print("\n[*] Evaluating best model...")
    evaluator = Evaluator(model, device, class_names)
    load_model(model)
    eval_results = evaluator.evaluate(val_loader)
    
    print(f"\n[✓] Best validation accuracy: {eval_results['top1_accuracy']:.2f}%")
    evaluator.print_classification_report(eval_results)
    
    # Save plots
    print("\n[*] Generating visualizations...")
    evaluator.plot_training_history(history, config.RESULTS_DIR / "training_history.png")
    evaluator.plot_confusion_matrix(eval_results, config.RESULTS_DIR / "confusion_matrix.png")
    print(f"[✓] Visualizations saved to {config.RESULTS_DIR}")


def evaluate(args):
    """Evaluation pipeline"""
    
    set_seed(config.SEED)
    device = get_device()
    
    # Get data
    print("\n[*] Loading CIFAR-10 dataset...")
    _, val_loader, class_names = get_data_loaders(batch_size=config.BATCH_SIZE)
    print(f"[✓] Validation set loaded")
    
    # Create model
    print("\n[*] Creating model...")
    model = resnet18(num_classes=config.NUM_CLASSES)
    print(f"[✓] Model created")
    
    # Load trained model
    if not load_model(model, "best_model.pt"):
        print("[-] No trained model found. Please train first.")
        return
    
    # Evaluate
    print("\n[*] Evaluating model...")
    evaluator = Evaluator(model, device, class_names)
    eval_results = evaluator.evaluate(val_loader)
    
    print(f"\n[✓] Evaluation Results:")
    print(f"    Top-1 Accuracy: {eval_results['top1_accuracy']:.2f}%")
    print(f"    Top-5 Accuracy: {eval_results['top5_accuracy']:.2f}%")
    
    evaluator.print_classification_report(eval_results)


def main():
    parser = argparse.ArgumentParser(
        description="Computer Vision CNN - CIFAR-10 Classification"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--model', type=str, choices=['resnet18', 'simplecnn'], 
                             default='resnet18', help='Model architecture')
    train_parser.add_argument('--epochs', type=int, default=config.NUM_EPOCHS,
                             help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE,
                             help='Batch size')
    train_parser.add_argument('--lr', type=float, default=config.LEARNING_RATE,
                             help='Learning rate')
    
    # Evaluation command
    eval_parser = subparsers.add_parser('eval', help='Evaluate the model')
    
    args = parser.parse_args()
    
    # Create output directories
    config.MODEL_DIR.mkdir(exist_ok=True)
    config.RESULTS_DIR.mkdir(exist_ok=True)
    
    if args.command == 'train':
        train(args)
    elif args.command == 'eval':
        evaluate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
