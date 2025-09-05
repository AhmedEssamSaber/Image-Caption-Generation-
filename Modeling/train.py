import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from utils import save_loss_plot, save_metrics_plots
from eval import evaluate_model
from config import LOSS_PLOTS_DIR, METRICS_PLOTS_DIR, SAVE_DIR

def train_model(encoder, decoder, train_loader, val_loader, train_idx, val_idx, dataset, vocab, device, num_epochs=10, lr=1e-4):
    encoder, decoder = encoder.to(device), decoder.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.pad_idx)
    params = list(decoder.parameters()) + list(encoder.cnn.fc.parameters()) + list(encoder.bn.parameters())
    optimizer = optim.Adam(params, lr=lr)

    train_losses, val_losses, all_metrics = [], [], []
    best_cider = -1
    ckpt_path = os.path.join(SAVE_DIR, "best_model.pt")

    for epoch in range(1, num_epochs+1):
        encoder.train(); decoder.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch}/{num_epochs}]")
        for imgs, caps, _ in loop:
            imgs, caps = imgs.to(device), caps.to(device)
            feats = encoder(imgs)
            outputs = decoder(feats, caps)
            targets = caps[:, 1:]
            loss = criterion(outputs.reshape(-1, outputs.size(2)), targets.reshape(-1))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        val_loss, refs, hyps = 0.0, [], []
        encoder.eval(); decoder.eval()
        with torch.no_grad():
            for imgs, caps, _ in val_loader:
                imgs, caps = imgs.to(device), caps.to(device)
                feats = encoder(imgs)
                outputs = decoder(feats, caps)
                targets = caps[:, 1:]
                loss = criterion(outputs.reshape(-1, outputs.size(2)), targets.reshape(-1))
                val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

        # Metrics
        metrics = evaluate_model(encoder, decoder, val_loader, vocab, device, max_len=30, limit=200)
        all_metrics.append(metrics)

        print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Metrics={metrics}")

        # Save best model
        if metrics["CIDEr"] > best_cider:
            best_cider = metrics["CIDEr"]
            torch.save({"encoder": encoder.state_dict(), "decoder": decoder.state_dict(), "epoch": epoch}, ckpt_path)

        # Save plots
        save_loss_plot(train_losses, val_losses, LOSS_PLOTS_DIR)
        save_metrics_plots(all_metrics, METRICS_PLOTS_DIR)

    return train_losses, val_losses, all_metrics
