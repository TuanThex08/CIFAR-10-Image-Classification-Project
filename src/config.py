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
    BATCH_SIZE: int = 64
    LEARNING_RATE: float = 0.1
    MOMENTUM: float = 0.9
    WEIGHT_DECAY: float = 5e-4
    NUM_EPOCHS: int = 5
    WARMUP_EPOCHS: int = 1
    
    # Optimizer & Scheduler
    SCHEDULER_TYPE: str = "cosine"  # "cosine", "step", "linear"
    STEP_SIZE: int = 50
    GAMMA: float = 0.1
    
    # Training setup
    DEVICE: str = "cpu"  # "cuda" or "cpu"
    SEED: int = 42
    NUM_WORKERS: int = 0
    PIN_MEMORY: bool = False
    
    # Data augmentation
    AUGMENTATION_ENABLED: bool = False
    CUTOUT_ENABLED: bool = False
    CUTOUT_SIZE: int = 8
    
    # Model checkpoint
    CHECKPOINT_INTERVAL: int = 10
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
