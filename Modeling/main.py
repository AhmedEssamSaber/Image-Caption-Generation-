from config import *
from utils import set_seed, split_by_image_indices
from dataset import Flickr30kDataset, MyCollate
from models import EncoderCNN, DecoderLSTM
from train import train_model
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader, Subset

def main():
    set_seed(SEED)
    transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    dataset = Flickr30kDataset(CSV_FILE, IMAGES_DIR, transform=transform, freq_threshold=FREQ_THRESHOLD)
    train_idx, val_idx = split_by_image_indices(dataset)
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True, collate_fn=MyCollate(dataset.pad_idx))
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False, collate_fn=MyCollate(dataset.pad_idx))
    encoder = EncoderCNN(EMBED_SIZE, dropout=DROPOUT)
    decoder = DecoderLSTM(EMBED_SIZE, HIDDEN_SIZE, vocab_size=len(dataset.vocab))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(encoder, decoder, train_loader, val_loader, train_idx, val_idx, dataset, dataset.vocab, device)

if __name__ == "__main__":
    main()
