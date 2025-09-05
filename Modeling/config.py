import os
import torch

# Paths
CSV_FILE   = "/kaggle/input/flickr-image-dataset/flickr30k_images/results.csv"
IMAGES_DIR = "/kaggle/input/flickr-image-dataset/flickr30k_images/flickr30k_images"
SAVE_DIR   = "/kaggle/working/ImgCap_results/"

# Hyperparameters
BATCH_SIZE = 32
EMBED_SIZE = 512
HIDDEN_SIZE = 512
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
FREQ_THRESHOLD = 5
DROPOUT = 0.3
NUM_WORKERS = 2 if torch.cuda.is_available() else 0
SEED = 42
VIS_EXAMPLES_PER_EPOCH = 8

# Directories
PLOTS_DIR = os.path.join(SAVE_DIR, "plots")
LOSS_PLOTS_DIR = os.path.join(PLOTS_DIR, "loss")
METRICS_PLOTS_DIR = os.path.join(PLOTS_DIR, "metrics")
VIS_DIR = os.path.join(SAVE_DIR, "visualizations")
METRICS_TEXT_DIR = os.path.join(SAVE_DIR, "metrics_txt")

for d in (PLOTS_DIR, LOSS_PLOTS_DIR, METRICS_PLOTS_DIR, VIS_DIR, METRICS_TEXT_DIR):
    os.makedirs(d, exist_ok=True)
