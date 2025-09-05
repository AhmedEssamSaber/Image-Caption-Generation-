import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pickle
from collections import Counter
from typing import List
import os

# ----------------- Model Architecture -----------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]

class ViTEncoder(nn.Module):
    def __init__(self, pretrained: bool = True, fine_tune_last_layer: bool = True):
        super().__init__()
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        vit = models.vit_b_16(weights=weights)
        self.patch_embed = vit.conv_proj
        self.encoder = vit.encoder
        self.cls_token = vit.class_token
        self.pos_embed = vit.encoder.pos_embedding
        self.hidden_dim = vit.hidden_dim
        
        # Freeze parameters initially
        for p in self.parameters():
            p.requires_grad = False
            
        # Optionally unfreeze the last encoder block
        if fine_tune_last_layer:
            for p in self.encoder.layers[-1].parameters():
                p.requires_grad = True

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        B, N, D = x.shape
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.encoder(x)
        tokens = x[:, 1:, :]  # drop CLS token
        return tokens

class TransformerCaptionDecoder(nn.Module):
    def __init__(self, feature_size: int, embed_size: int, vocab_size: int,
                 num_layers: int = 3, nhead: int = 8, ff_dim: int = 2048, 
                 dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.pos_enc = PositionalEncoding(embed_size, max_len=max_len)
        self.feat_proj = nn.Linear(feature_size, embed_size)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size, nhead=nhead, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, captions):
        # features: (B, N, feature_size)
        # captions: (B, T)
        tgt = self.embed(captions)
        tgt = self.pos_enc(tgt)
        memory = self.feat_proj(features)
        
        # Causal mask for autoregressive decoding
        T = captions.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(captions.device)
        
        out = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)
        logits = self.fc(out)
        return logits

class ImgCapViTTransformer(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int = 512, num_layers: int = 3,
                 nhead: int = 8, ff_dim: int = 2048, dropout: float = 0.1, 
                 fine_tune_last_layer: bool = True):
        super().__init__()
        self.vit = ViTEncoder(pretrained=True, fine_tune_last_layer=fine_tune_last_layer)
        self.decoder = TransformerCaptionDecoder(
            feature_size=self.vit.hidden_dim,
            embed_size=embed_size,
            vocab_size=vocab_size,
            num_layers=num_layers,
            nhead=nhead,
            ff_dim=ff_dim,
            dropout=dropout,
            max_len=1000
        )

    def forward(self, images, captions):
        feats = self.vit(images)
        logits = self.decoder(feats, captions[:, :-1])  # teacher forcing
        return logits

    @torch.no_grad()
    def caption_image(self, image, vocabulary, max_len: int = 30):
        self.eval()
        device = next(self.parameters()).device
        feats = self.vit(image)
        memory = self.decoder.feat_proj(feats)
        
        sos_idx = vocabulary.stoi["<SOS>"]
        eos_idx = vocabulary.stoi["<EOS>"]
        
        # Start with SOS token
        ys = torch.tensor([[sos_idx]], dtype=torch.long, device=device)
        
        for _ in range(max_len):
            tgt = self.decoder.embed(ys)
            tgt = self.decoder.pos_enc(tgt)
            
            T = ys.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(device)
            
            out = self.decoder.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)
            logits = self.decoder.fc(out[:, -1, :])
            next_token = torch.argmax(logits, dim=1).unsqueeze(1)
            
            ys = torch.cat([ys, next_token], dim=1)
            
            # Stop if EOS is generated
            if next_token.item() == eos_idx:
                break
        
        # Convert indices to words
        caption_indices = ys.squeeze(0).cpu().tolist()
        caption = vocabulary.decode(caption_indices)
        return caption

# ----------------- Vocabulary Class -----------------
class Vocabulary:
    def __init__(self, freq_threshold: int = 5):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer(text: str):
        return [t for t in text.lower().strip().split() if t]

    def build_vocabulary(self, sentence_list: List[str]):
        frequencies = Counter()
        for sentence in sentence_list:
            if not isinstance(sentence, str):
                continue
            for token in self.tokenizer(sentence):
                frequencies[token] += 1

        idx = max(self.itos.keys()) + 1
        for word, freq in frequencies.items():
            if freq >= self.freq_threshold and word not in self.stoi:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text: str) -> List[int]:
        tokens = self.tokenizer(text)
        return [self.stoi.get(tok, self.stoi["<UNK>"]) for tok in tokens]

    def decode(self, indices: List[int]) -> str:
        words = []
        for idx in indices:
            if idx == self.stoi["<EOS>"]:
                break
            if idx in (self.stoi["<PAD>"], self.stoi["<SOS>"]):
                continue
            words.append(self.itos.get(idx, "<UNK>"))
        return " ".join(words) if words else "<UNK>"

# ----------------- Cached Loaders -----------------
@st.cache_resource
def load_vocab(vocab_path):
    try:
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
        return vocab
    except FileNotFoundError:
        st.error(f"Vocabulary file not found at {vocab_path}")
        return None

@st.cache_resource
def load_model(model_path, vocab_size, device="cpu"):
    model = ImgCapViTTransformer(
        vocab_size=vocab_size,
        embed_size=512,
        num_layers=3,
        nhead=8,
        ff_dim=2048,
        dropout=0.1
    )
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at {model_path}")
        return None

# ----------------- Preprocessing -----------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def preprocess(image: Image.Image):
    return transform(image).unsqueeze(0)

# ----------------- Streamlit UI -----------------
st.title("🖼️ Image Captioning with ViT + Transformer")
st.write("Upload an image and let the model generate a caption.")

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Try to find model and vocab files with different possible paths
possible_vocab_paths = [
    os.path.join(current_dir, "vocab.pkl"),
    "vocab.pkl",
    r"D:\Ai courses\projects\VIT\vocab.pkl"
]

possible_model_paths = [
    os.path.join(current_dir, "best_by_cider.pth"),
    "best_by_cider.pth",
    r"D:\Ai courses\projects\VIT\best_by_cider.pth"
]

vocab_path = None
model_path = None

for path in possible_vocab_paths:
    if os.path.exists(path):
        vocab_path = path
        break

for path in possible_model_paths:
    if os.path.exists(path):
        model_path = path
        break

if vocab_path is None:
    st.error("Vocabulary file not found. Please make sure vocab.pkl is in the correct location.")
    st.stop()

if model_path is None:
    st.error("Model file not found. Please make sure best_by_cider.pth is in the correct location.")
    st.stop()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.write(f"Using device: {device}")
    
    # Load vocabulary and model
    vocab = load_vocab(vocab_path)
    if vocab is None:
        st.error("Failed to load vocabulary.")
        st.stop()
        
    model = load_model(model_path, len(vocab), device)
    if model is None:
        st.error("Failed to load model.")
        st.stop()
    
    # Preprocess image and generate caption
    image_tensor = preprocess(image).to(device)
    
    with st.spinner("Generating caption..."):
        caption = model.caption_image(image_tensor, vocab)
    
    st.subheader("📝 Generated Caption")
    st.success(caption)