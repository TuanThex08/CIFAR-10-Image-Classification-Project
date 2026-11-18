"""
Data loading and augmentation pipeline
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.config import config


class DataAugmentation:
    """Data augmentation transformations"""
    
    @staticmethod
    def get_train_transform():
        """Training data augmentation"""
        if not config.AUGMENTATION_ENABLED:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2023, 0.1994, 0.2010)
                ),
            ])
        
        return transforms.Compose([
            transforms.RandomCrop(config.IMAGE_SIZE, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010)
            ),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.1)) if config.CUTOUT_ENABLED else transforms.Lambda(lambda x: x),
        ])
    
    @staticmethod
    def get_test_transform():
        """Test data transformations (no augmentation)"""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010)
            ),
        ])


class CIFARDataModule:
    """Data module for CIFAR-10 dataset"""
    
    def __init__(self, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_aug = DataAugmentation()
    
    def get_train_loader(self):
        """Get training data loader"""
        train_dataset = datasets.CIFAR10(
            root=str(config.DATA_DIR),
            train=True,
            download=True,
            transform=self.data_aug.get_train_transform()
        )
        
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=config.PIN_MEMORY,
            drop_last=True
        )
    
    def get_val_loader(self):
        """Get validation data loader"""
        val_dataset = datasets.CIFAR10(
            root=str(config.DATA_DIR),
            train=False,
            download=True,
            transform=self.data_aug.get_test_transform()
        )
        
        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=config.PIN_MEMORY
        )
    
    def get_class_names(self):
        """Get CIFAR-10 class names"""
        return ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def get_data_loaders(batch_size=config.BATCH_SIZE):
    """Utility function to get train and val loaders"""
    data_module = CIFARDataModule(batch_size=batch_size)
    return data_module.get_train_loader(), data_module.get_val_loader(), data_module.get_class_names()
