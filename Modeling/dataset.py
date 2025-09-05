import os, pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from vocab import Vocabulary

class Flickr30kDataset(Dataset):
    def __init__(self, csv_file, images_dir, transform=None, freq_threshold=5):
        self.images_dir = images_dir
        self.transform = transform
        df = pd.read_csv(csv_file, sep="|", engine="python")
        df.columns = df.columns.str.strip()
        df = df.rename(columns={"comment": "caption", "image_name": "image"})
        self.df = df[["image", "caption"]].dropna().reset_index(drop=True)
        self.image_to_captions = df.groupby("image")["caption"].apply(list).to_dict()
        self.vocab = Vocabulary(freq_threshold=freq_threshold)
        self.vocab.build_vocabulary(self.df["caption"].tolist())
        self.pad_idx = self.vocab.stoi["<PAD>"]
        self.sos_idx = self.vocab.stoi["<SOS>"]
        self.eos_idx = self.vocab.stoi["<EOS>"]

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_dir, row["image"])
        image = Image.open(img_path).convert("RGB")
        if self.transform: image = self.transform(image)
        caption_ids = [self.sos_idx] + self.vocab.numericalize(row["caption"]) + [self.eos_idx]
        return image, torch.tensor(caption_ids), row["image"]

class MyCollate:
    def __init__(self, pad_idx): self.pad_idx = pad_idx
    def __call__(self, batch):
        imgs = torch.stack([x[0] for x in batch])
        caps = pad_sequence([x[1] for x in batch], batch_first=True, padding_value=self.pad_idx)
        names = [x[2] for x in batch]
        return imgs, caps, names
