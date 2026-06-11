# config.py
import pathlib


class Config:
    # Paths
    DATASET_PATH = pathlib.Path("data/processed")

    # Training parameters
    N_EPOCHS_AE = 600
    N_EPOCHS_DNN = 700
    TEST_SIZE = 0.4
    VAL_SIZE = 0.33
    LEARNING_RATE_AE = 1e-3

    # BEST VAL FOR ONLY REGULAR
    # batch_size = 16
    # lr = 0.005
    # weight_decay = 0.0001
    # BEST VAL FOR ALL CLUSTERS
    # batch_size = 8
    # lr = 0.005
    # weight_decay = 0.01
    BATCH_SIZE = 8
    LEARNING_RATE_DNN = 0.005
    WEIGHT_DECAY = 0.01

    # Hyperparameter tuning for LR, Batch Size and Weight Decay
    # BEST VAL FOR ONLY REGULAR
    # lr = 0.0005
    # max_depth = 4
    LEARNING_RATE_XG = 0.0005
    MAX_DEPTH = 4
    # Model parameters
    EMBEDDING_DIM = 16

    # Reproducibility
    SEED = 42

    ONLY_REGULAR = True
