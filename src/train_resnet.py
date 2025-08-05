import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from config import DEVICE, EPOCHS, LR, MODEL_DIR, RESULTS_DIR
from data_utils import get_dataloaders
from model_utils import get_model
from evaluate import evaluate_model

def train():
    train_loader, val_loader, class_names = get_dataloaders()
    model = get_model().to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss, correct, total = 0, 0, 0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")

        for imgs, labels in loop:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            epoch_loss += loss.item()

            loop.set_postfix(loss=loss.item(), acc=100 * correct / total)

        torch.save(model.state_dict(), f"{MODEL_DIR}/resnet_epoch{epoch+1}.pth")

    acc, f1 = evaluate_model(model, val_loader, class_names)
    with open(f"{RESULTS_DIR}/metrics.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\nF1 Score: {f1:.4f}\n")

if __name__ == "__main__":
    train()
