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
            drop_last=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None
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
            pin_memory=config.PIN_MEMORY,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None
        )
    
    def get_class_names(self):
        """Get CIFAR-10 class names"""
        return ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)


class GPUDataLoader:
    """Whole-dataset-on-GPU loader for CIFAR-10.

    CIFAR-10 is only ~180MB as uint8, so it fits in VRAM. Keeping it there removes
    PIL decoding, worker processes (slow to spawn on Windows) and host->device
    copies. Random crop (pad 4) + horizontal flip are done on the GPU per batch.
    """

    def __init__(self, images, labels, batch_size, device, train, augment, drop_last):
        # images: uint8 numpy (N, 32, 32, 3) kept as uint8 NCHW on GPU (~150MB for
        # CIFAR-10 train); each batch is normalized to fp32 when yielded. Large
        # one-shot float allocations can OOM on 4GB Windows (WDDM) GPUs.
        self.images = torch.from_numpy(images).to(device).permute(0, 3, 1, 2).contiguous()
        self.mean = torch.tensor(CIFAR_MEAN, device=device).view(1, 3, 1, 1) * 255
        self.std = torch.tensor(CIFAR_STD, device=device).view(1, 3, 1, 1) * 255
        self.labels = torch.as_tensor(labels, dtype=torch.long, device=device)
        self.batch_size = batch_size
        self.device = device
        self.train = train
        self.augment = augment
        self.drop_last = drop_last
        self.n = self.labels.numel()

    def __len__(self):
        if self.drop_last:
            return self.n // self.batch_size
        return (self.n + self.batch_size - 1) // self.batch_size

    def _augment(self, x):
        b = x.size(0)
        # Random horizontal flip
        flip = torch.rand(b, device=self.device) < 0.5
        x = torch.where(flip.view(b, 1, 1, 1), x.flip(3), x)
        # Random crop 32x32 from zero-padded 40x40 (same as RandomCrop(32, padding=4))
        pad = 4
        xp = torch.nn.functional.pad(x, (pad, pad, pad, pad))
        dx = torch.randint(0, 2 * pad + 1, (b,), device=self.device)
        dy = torch.randint(0, 2 * pad + 1, (b,), device=self.device)
        ar = torch.arange(32, device=self.device)
        rows = (dy.view(b, 1) + ar.view(1, 32))  # (b, 32)
        cols = (dx.view(b, 1) + ar.view(1, 32))  # (b, 32)
        bi = torch.arange(b, device=self.device).view(b, 1, 1)
        # gather -> (b, 32, 32, C) then back to NCHW
        xp = xp.permute(0, 2, 3, 1)
        out = xp[bi, rows.view(b, 32, 1), cols.view(b, 1, 32)]
        return out.permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)

    def __iter__(self):
        if self.train:
            idx = torch.randperm(self.n, device=self.device)
        else:
            idx = torch.arange(self.n, device=self.device)
        for i in range(len(self)):
            j = idx[i * self.batch_size:(i + 1) * self.batch_size]
            x = ((self.images[j].float() - self.mean) / self.std).contiguous(
                memory_format=torch.channels_last)
            if self.augment:
                x = self._augment(x)
            yield x, self.labels[j]


def get_gpu_data_loaders(batch_size, device):
    train_ds = datasets.CIFAR10(root=str(config.DATA_DIR), train=True, download=True)
    val_ds = datasets.CIFAR10(root=str(config.DATA_DIR), train=False, download=True)
    train_loader = GPUDataLoader(train_ds.data, train_ds.targets, batch_size, device,
                                 train=True, augment=config.AUGMENTATION_ENABLED, drop_last=True)
    val_loader = GPUDataLoader(val_ds.data, val_ds.targets, max(batch_size, 1024), device,
                               train=False, augment=False, drop_last=False)
    return train_loader, val_loader


def get_data_loaders(batch_size=config.BATCH_SIZE):
    """Utility function to get train and val loaders"""
    data_module = CIFARDataModule(batch_size=batch_size)
    if config.GPU_DATA and config.DEVICE == "cuda" and torch.cuda.is_available():
        train_loader, val_loader = get_gpu_data_loaders(batch_size, torch.device("cuda"))
        return train_loader, val_loader, data_module.get_class_names()
    return data_module.get_train_loader(), data_module.get_val_loader(), data_module.get_class_names()
