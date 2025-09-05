import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


# -------------------------------
# EncoderCNN
# -------------------------------
class EncoderCNN(nn.Module):
    def __init__(self, encoded_image_size=14, encoder_dim=2048):
        super(EncoderCNN, self).__init__()
        self.enc_image_size = encoded_image_size
        self.encoder_dim = encoder_dim

        # Load pre-trained ResNet50
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Remove linear and pool layers (fc + avgpool)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Resize output to fixed size (14x14)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        # Freeze batchnorm layers
        for module in self.resnet.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    def forward(self, images):
        features = self.resnet(images)  # (batch_size, 2048, H/32, W/32)
        features = self.adaptive_pool(features)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        features = features.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return features


# -------------------------------
# Attention Mechanism
# -------------------------------
class Attention(nn.Module):
    def __init__(self, encoder_dim, hidden_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # Linear layer to transform encoder's output
        self.decoder_att = nn.Linear(hidden_dim, attention_dim)   # Linear layer to transform decoder's hidden state
        self.full_att = nn.Linear(attention_dim, 1)              # Linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # Softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        # encoder_out: (batch_size, num_pixels, encoder_dim)
        # decoder_hidden: (batch_size, hidden_dim)

        att1 = self.encoder_att(encoder_out)      # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)   # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)                 # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


# -------------------------------
# Decoder with Attention
# -------------------------------
class DecoderWithAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim, encoder_dim, hidden_dim, attention_dim, dropout=0.3):
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.attention = Attention(encoder_dim, hidden_dim, attention_dim)

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(p=dropout)

        # LSTMCell
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, hidden_dim, bias=True)

        # Linear layers to initialize LSTM state
        self.init_h = nn.Linear(encoder_dim, hidden_dim)
        self.init_c = nn.Linear(encoder_dim, hidden_dim)

        # Linear layer to create a sigmoid-activated gate
        self.f_beta = nn.Linear(hidden_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()

        # Final output layer
        self.fc = nn.Linear(hidden_dim, vocab_size)

        self.init_weights()

    def init_weights(self):
        """Initialize embedding and fc layer with uniform distribution."""
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        """Initialize hidden and cell states of LSTM based on encoder output."""
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, hidden_dim)
        c = self.init_c(mean_encoder_out)  # (batch_size, hidden_dim)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation.
        :param encoder_out: encoded images, (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, (batch_size, 1)
        """
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)

        # We won't decode at the <end> position, since we've finished generating already
        decode_lengths = (caption_lengths - 1).tolist()

        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(encoder_out.device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(encoder_out.device)

        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])

            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar
            attention_weighted_encoding = gate * attention_weighted_encoding

            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t])
            )  # (batch_size_t, hidden_dim)

            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind
