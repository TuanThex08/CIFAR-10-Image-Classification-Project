"""
Training module for CNN model
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, LinearLR
import os
import time
from pathlib import Path
from src.config import config
from src.utils import accuracy, save_checkpoint, save_best_model


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

        # Mixed precision (AMP) - only effective on CUDA
        self.use_amp = (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # GPU speed optimizations (fixed 32x32 input -> cudnn can autotune)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            # TF32 matmul/conv on Ampere+ (free speedup, negligible accuracy impact)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            self.model = self.model.to(memory_format=torch.channels_last)

        # torch.compile (PyTorch 2.x) - fuses kernels, big speedup after warmup.
        # Inductor backend needs Triton (not available on stock Windows), and the
        # failure only surfaces on the first forward pass, so gate on Triton here.
        if config.COMPILE_MODEL and hasattr(torch, "compile"):
            # Default cache under %TEMP% exceeds Windows MAX_PATH (260) -> FileNotFoundError.
            # Use a short path. Cached kernels make later runs start in ~10s instead of ~60s.
            if config.COMPILE_CACHE_DIR:
                os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", config.COMPILE_CACHE_DIR)
                os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(config.COMPILE_CACHE_DIR, "triton"))
            try:
                import triton  # noqa: F401
                self.model = torch.compile(self.model)
                print("[Trainer] torch.compile enabled")
            except Exception as e:
                print(f"[Trainer] torch.compile skipped ({type(e).__name__}: "
                      f"Triton not available); continuing without it")

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

        # Accumulate metrics on-device to avoid a host-device sync every batch;
        # only pull scalars to CPU (.item()) when logging and at epoch end.
        loss_sum = torch.zeros((), device=self.device)
        top1_sum = torch.zeros((), device=self.device)
        top5_sum = torch.zeros((), device=self.device)
        n_samples = 0

        start_time = time.time()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            if self.use_amp:
                inputs = inputs.to(memory_format=torch.channels_last)

            # Forward pass (autocast for mixed precision on GPU)
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            # Backward pass (scaled for AMP numerical stability)
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Calculate accuracy (kept on-device; no .item() here)
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))

            bs = inputs.size(0)
            loss_sum += loss.detach() * bs
            top1_sum += acc1[0] * bs
            top5_sum += acc5[0] * bs
            n_samples += bs

            if (batch_idx + 1) % config.LOG_INTERVAL == 0:
                elapsed_time = time.time() - start_time
                avg_loss = (loss_sum / n_samples).item()
                avg_top1 = (top1_sum / n_samples).item()
                avg_top5 = (top5_sum / n_samples).item()
                print(f"[Epoch {epoch}] [{batch_idx + 1}/{len(train_loader)}] "
                      f"Loss: {avg_loss:.4f} | Top-1: {avg_top1:.2f}% | "
                      f"Top-5: {avg_top5:.2f}% | Time: {elapsed_time:.1f}s")

        avg_loss = (loss_sum / n_samples).item()
        avg_top1 = (top1_sum / n_samples).item()
        self.train_losses.append(avg_loss)
        self.train_accuracies.append(avg_top1)

        return avg_loss, avg_top1
    
    def validate(self, val_loader, epoch):
        """Validate the model"""
        self.model.eval()
        
        loss_sum = torch.zeros((), device=self.device)
        top1_sum = torch.zeros((), device=self.device)
        top5_sum = torch.zeros((), device=self.device)
        n_samples = 0

        with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=self.use_amp):
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                if self.use_amp:
                    inputs = inputs.to(memory_format=torch.channels_last)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))

                bs = inputs.size(0)
                loss_sum += loss.detach() * bs
                top1_sum += acc1[0] * bs
                top5_sum += acc5[0] * bs
                n_samples += bs

        avg_loss = (loss_sum / n_samples).item()
        avg_top1 = (top1_sum / n_samples).item()
        avg_top5 = (top5_sum / n_samples).item()
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(avg_top1)

        print(f"[Validation] Loss: {avg_loss:.4f} | Top-1: {avg_top1:.2f}% | Top-5: {avg_top5:.2f}%")

        return avg_loss, avg_top1
    
    def _raw_model(self):
        # torch.compile wraps the model; save/load the underlying module's weights
        return getattr(self.model, "_orig_mod", self.model)

    def save_resume_state(self, epoch, filename="checkpoint.pt"):
        """Save everything needed to continue training exactly where it stopped."""
        state = {
            'epoch': epoch,
            'model_state_dict': self._raw_model().state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_acc': self.best_acc,
            'patience_counter': self.patience_counter,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'cpu_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        path = config.MODEL_DIR / filename
        tmp = path.with_suffix(".tmp")
        torch.save(state, tmp)
        tmp.replace(path)  # atomic: an interrupt mid-save never corrupts the old checkpoint

    def load_resume_state(self, filename="checkpoint.pt"):
        """Restore state from a checkpoint. Returns the epoch to start from."""
        path = config.MODEL_DIR / filename
        if not path.exists():
            print(f"[Resume] No checkpoint at {path}; starting from scratch")
            return 1
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self._raw_model().load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' not in ckpt:
            # Old-format checkpoint (model + optimizer only): rebuild LR position
            print("[Resume] Old checkpoint format: no scheduler/history; fast-forwarding LR")
            for _ in range(ckpt['epoch']):
                self.scheduler.step()
            return ckpt['epoch'] + 1
        self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.scaler.load_state_dict(ckpt['scaler_state_dict'])
        self.best_acc = ckpt['best_acc']
        self.patience_counter = ckpt['patience_counter']
        self.train_losses = ckpt['train_losses']
        self.val_losses = ckpt['val_losses']
        self.train_accuracies = ckpt['train_accuracies']
        self.val_accuracies = ckpt['val_accuracies']
        torch.set_rng_state(ckpt['cpu_rng_state'].cpu())
        if ckpt.get('cuda_rng_state') is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in ckpt['cuda_rng_state']])
        print(f"[Resume] Loaded {path} (epoch {ckpt['epoch']}, best acc {self.best_acc:.2f}%)")
        return ckpt['epoch'] + 1

    def train(self, train_loader, val_loader, start_epoch=1):
        """Full training loop"""
        print(f"Starting training for {config.NUM_EPOCHS} epochs (from epoch {start_epoch})...")
        print(f"Number of trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

        for epoch in range(start_epoch, config.NUM_EPOCHS + 1):
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
            
            # Save best model
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.patience_counter = 0
                best_metrics = {'accuracy': val_acc, 'loss': val_loss}
                save_best_model(self._raw_model(), best_metrics)
            else:
                self.patience_counter += 1

            # Resume checkpoint (after best/patience update so the state is consistent)
            if epoch % config.CHECKPOINT_INTERVAL == 0 or epoch == config.NUM_EPOCHS:
                self.save_resume_state(epoch)
                print(f"Checkpoint saved (epoch {epoch})")

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
