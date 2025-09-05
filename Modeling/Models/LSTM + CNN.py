import torch, torch.nn as nn
import torchvision.models as models
from vocab import Vocabulary

class EncoderCNN(nn.Module):
    def __init__(self, embed_size, dropout=0.3, train_cnn=False):
        super().__init__()
        self.cnn = models.resnet50(weights="IMAGENET1K_V2")
        for p in self.cnn.parameters(): p.requires_grad = train_cnn
        in_feats = self.cnn.fc.in_features
        self.cnn.fc = nn.Sequential(
            nn.Linear(in_feats, embed_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_size, embed_size)
        )
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
    def forward(self, x): return self.bn(self.cnn(x))

class DecoderLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.init_h = nn.Linear(embed_size, hidden_size)
        self.init_c = nn.Linear(embed_size, hidden_size)

    def init_hidden(self, feat):
        h = self.init_h(feat).unsqueeze(0)
        c = self.init_c(feat).unsqueeze(0)
        return h, c

    def forward(self, feats, captions):
        h, c = self.init_hidden(feats)
        emb = self.embed(captions[:, :-1])
        out, _ = self.lstm(emb, (h, c))
        return self.fc(out)

    def greedy_generate(self, feat, vocab: Vocabulary, max_len=30):
        self.eval()
        with torch.no_grad():
            h, c = self.init_hidden(feat)
            word = torch.tensor([[vocab.stoi["<SOS>"]]], device=feat.device)
            result = []
            for _ in range(max_len):
                emb = self.embed(word)
                out, (h, c) = self.lstm(emb, (h, c))
                logits = self.fc(out.squeeze(1))
                word = torch.argmax(logits, dim=1, keepdim=True)
                result.append(word.item())
                if word.item() == vocab.stoi["<EOS>"]: break
        return vocab.decode(result)
