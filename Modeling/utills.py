import os, random
import numpy as np
import torch
import matplotlib.pyplot as plt

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_loss_plot(train_losses, val_losses, save_dir):
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "loss.png"), dpi=150, bbox_inches="tight")
    plt.close()

def save_metrics_plots(all_metrics, save_dir):
    import matplotlib.pyplot as plt
    epochs = range(1, len(all_metrics)+1)
    plt.figure(figsize=(10,6))
    for m in ["BLEU-1","BLEU-2","BLEU-3","BLEU-4"]:
        plt.plot(epochs, [x[m] for x in all_metrics], label=m)
    plt.xlabel("epoch"); plt.ylabel("BLEU"); plt.legend()
    plt.savefig(os.path.join(save_dir, "bleu.png"))
    plt.close()

def split_by_image_indices(dataset, train_ratio=0.8, seed=42):
    unique_images = list(dataset.image_to_captions.keys())
    random.Random(seed).shuffle(unique_images)
    split = int(train_ratio * len(unique_images))
    train_images = set(unique_images[:split])
    train_indices = [i for i, row in dataset.df.iterrows() if row['image_name'] in train_images]
    val_indices = [i for i, row in dataset.df.iterrows() if row['image_name'] not in train_images]
    return train_indices, val_indices
