from collections import Counter
from typing import List

class Vocabulary:
    def __init__(self, freq_threshold: int = 5):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}

    def __len__(self): return len(self.itos)

    @staticmethod
    def tokenizer(text: str): return [t for t in text.lower().strip().split() if t]

    def build_vocabulary(self, sentence_list: List[str]):
        frequencies = Counter()
        for sentence in sentence_list:
            if isinstance(sentence, str):
                for token in self.tokenizer(sentence):
                    frequencies[token] += 1
        idx = max(self.itos.keys()) + 1
        for word, freq in frequencies.items():
            if freq >= self.freq_threshold and word not in self.stoi:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text: str) -> List[int]:
        return [self.stoi.get(tok, self.stoi["<UNK>"]) for tok in self.tokenizer(text)]

    def decode(self, indices: List[int]) -> str:
        words = []
        for idx in indices:
            if idx == self.stoi["<EOS>"]: break
            if idx not in (self.stoi["<PAD>"], self.stoi["<SOS>"]):
                words.append(self.itos.get(idx, "<UNK>"))
        return " ".join(words) if words else "<UNK>"
