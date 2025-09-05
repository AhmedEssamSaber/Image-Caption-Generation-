import torch
import torch.nn as nn
import torchvision.models as models
import math

class ViTEncoder(nn.Module):
    def __init__(self, model_name="vit_b_16", pretrained=True):
        super(ViTEncoder, self).__init__()
        if model_name == "vit_b_16":
            vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
        else:
            raise ValueError(f"Unsupported ViT model: {model_name}")

        self.patch_embed = vit.conv_proj       # patch embedding
        self.cls_token = vit.class_token
        self.pos_embed = vit.encoder.pos_embedding
        self.encoder = vit.encoder            # transformer encoder
        self.hidden_size = vit.hidden_dim

    def forward(self, images):
        # Convert image to patch embeddings
        x = self.patch_embed(images)          # [B, hidden_dim, H/16, W/16]
        x = x.flatten(2).transpose(1, 2)      # [B, num_patches, hidden_dim]

        # Add class token
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional embeddings
        x = x + self.pos_embed

        # Pass through transformer encoder
        x = self.encoder(x)

        return x   # [B, num_patches+1, hidden_dim]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerCaptionDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, num_heads, dropout=0.1):
        super(TransformerCaptionDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = PositionalEncoding(embed_size)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

        if embed_size != hidden_size:
            self.embedding_proj = nn.Linear(embed_size, hidden_size)
        else:
            self.embedding_proj = nn.Identity()

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        tgt_embedded = self.embedding(tgt)
        tgt_embedded = self.embedding_proj(tgt_embedded)
        tgt_embedded = self.positional_encoding(tgt_embedded)

        tgt_embedded = tgt_embedded.transpose(0, 1)
        memory = memory.transpose(0, 1)

        output = self.transformer_decoder(
            tgt_embedded,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask
        )

        output = output.transpose(0, 1)
        return self.fc_out(output)

class ImgCapViTTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, num_heads, dropout=0.1):
        super(ImgCapViTTransformer, self).__init__()
        self.encoder = ViTEncoder()
        self.decoder = TransformerCaptionDecoder(vocab_size, embed_size, hidden_size, num_layers, num_heads, dropout)

    def forward(self, images, captions, tgt_mask=None):
        memory = self.encoder(images)
        outputs = self.decoder(captions, memory, tgt_mask=tgt_mask)
        return outputs

    def caption_image(self, image, vocab, max_len=20, device="cpu"):
        self.eval()
        with torch.no_grad():
            memory = self.encoder(image.to(device))
            outputs = [vocab.stoi["<SOS>"]]

            for _ in range(max_len):
                tgt = torch.tensor(outputs).unsqueeze(0).to(device)  # (1, seq_len)
                out = self.decoder(tgt, memory)
                pred = out[:, -1, :].argmax(dim=-1).item()
                outputs.append(pred)
                if pred == vocab.stoi["<EOS>"]:
                    break

        return vocab.decode(outputs)
