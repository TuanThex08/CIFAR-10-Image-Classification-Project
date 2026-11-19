"""
Configuration file for Computer Vision CNN Project
"""
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Config:
    """Main configuration class for the project"""
    
    # Project paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    MODEL_DIR: Path = PROJECT_ROOT / "models"
    RESULTS_DIR: Path = PROJECT_ROOT / "results"
    
    # Model architecture
    NUM_CLASSES: int = 10  # CIFAR-10
    INPUT_CHANNELS: int = 3
    IMAGE_SIZE: int = 32
    
    # Training parameters
    BATCH_SIZE: int = 256
    LEARNING_RATE: float = 0.1
    MOMENTUM: float = 0.9
    WEIGHT_DECAY: float = 5e-4
    NUM_EPOCHS: int = 200
    WARMUP_EPOCHS: int = 1
    
    # Optimizer & Scheduler
    SCHEDULER_TYPE: str = "cosine"  # "cosine", "step", "linear"
    STEP_SIZE: int = 50
    GAMMA: float = 0.1
    
    # Training setup
    DEVICE: str = "cuda"  # "cuda" or "cpu"
    # torch.compile: needs `triton-windows` 3.4.x for torch 2.8 (uv pip install "triton-windows<3.5,>=3.4").
    # Measured on RTX 3050 Laptop: ~8x faster per step; first epoch adds ~60s compile time (cached after).
    COMPILE_MODEL: bool = True
    COMPILE_CACHE_DIR: str = r"C:\tic"  # short path: default %TEMP% path breaks Windows MAX_PATH
    SEED: int = 42
    NUM_WORKERS: int = 4  # each worker commits ~1.6GB on Windows (torch+CUDA re-import)
    PIN_MEMORY: bool = True
    GPU_DATA: bool = True  # keep whole CIFAR-10 on GPU (no DataLoader workers / PIL)

    # Data augmentation (on GPU_DATA path: random crop + flip, essentially free)
    AUGMENTATION_ENABLED: bool = True
    CUTOUT_ENABLED: bool = False
    CUTOUT_SIZE: int = 8
    
    # Model checkpoint
    CHECKPOINT_INTERVAL: int = 1  # every epoch, so --resume loses at most one epoch
    EARLY_STOPPING_PATIENCE: int = 30
    
    # Logging
    LOG_INTERVAL: int = 200
    VERBOSE: bool = True
    
    def __post_init__(self):
        """Create necessary directories"""
        self.DATA_DIR.mkdir(exist_ok=True)
        self.MODEL_DIR.mkdir(exist_ok=True)
        self.RESULTS_DIR.mkdir(exist_ok=True)


# Global config instance
config = Config()
