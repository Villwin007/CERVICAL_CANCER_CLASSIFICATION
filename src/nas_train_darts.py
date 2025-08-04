import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import os

from nas_darts_model import DARTSNetwork
from nas_operations import OPS  # Make sure this is imported

def get_data(batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder('data/raw/cervicalCancer', transform=transform)
    train_len = int(0.8 * len(dataset))
    train_set, val_set = random_split(dataset, [train_len, len(dataset) - train_len])
    return DataLoader(train_set, batch_size=batch_size, shuffle=True), DataLoader(val_set, batch_size=batch_size)

def parse_genotype(model):
    genotype = []
    for alpha in model.alphas:
        probs = F.softmax(alpha, dim=0)
        top_op_idx = torch.argmax(probs).item()
        op_name = list(OPS.keys())[top_op_idx]
        genotype.append(op_name)
    return genotype

def evaluate(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print(f"Validation Accuracy: {acc:.2f}%")
    return acc

def train_darts(epochs=30):
    train_loader, val_loader = get_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DARTSNetwork().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")

    # Save final architecture and model
    genotype = parse_genotype(model)
    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/results", exist_ok=True)

    torch.save(model.state_dict(), "outputs/models/darts_final.pth")
    with open("outputs/results/darts_genotype.txt", "w") as f:
        f.write(str(genotype))
    print("Final architecture (genotype):", genotype)

    # Evaluate on validation set
    evaluate(model, val_loader, device)

if __name__ == '__main__':
    train_darts()
