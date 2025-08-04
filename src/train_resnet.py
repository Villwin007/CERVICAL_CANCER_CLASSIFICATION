import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
import numpy as np
from data_utils import get_dataloaders
from model_utils import get_model
from nas_darts_model import DARTSNetwork, genotype
from config import DEVICE, MODEL_DIR, RESULTS_DIR
import os

def evaluate(model, dataloader, class_names):
    model.eval()
    y_true, y_pred, y_probs = [], [], []

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            probs = nn.Softmax(dim=1)(outputs)
            preds = torch.argmax(probs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

    return np.array(y_true), np.array(y_pred), np.array(y_probs)

def plot_confusion_matrix(y_true, y_pred, class_names, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()

def plot_roc(y_true, y_probs, class_names, title, filename):
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_probs.ravel())
    roc_auc = roc_auc_score(y_true_bin, y_probs, average="macro", multi_class="ovr")

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()

def evaluate_model(model, model_name, model_path, val_loader, class_names):
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)

    start_time = time.time()
    y_true, y_pred, y_probs = evaluate(model, val_loader, class_names)
    inference_time = time.time() - start_time

    acc = np.mean(y_true == y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    f1 = float(report.split()[-2])  # macro avg f1-score

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6  # in MB

    print(f"\n🔍 {model_name} Evaluation:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Model Size: {model_size:.2f}M params")
    print(f"Inference Time: {inference_time:.2f}s")
    print(report)

    # Save results
    with open(os.path.join(RESULTS_DIR, f"{model_name}_report.txt"), "w") as f:
        f.write(f"{model_name} Results:\n\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"Model Size: {model_size:.2f}M params\n")
        f.write(f"Inference Time: {inference_time:.2f}s\n\n")
        f.write(report)

    plot_confusion_matrix(y_true, y_pred, class_names, f"{model_name} Confusion Matrix", f"{model_name}_confusion.png")
    plot_roc(y_true, y_probs, class_names, f"{model_name} ROC Curve", f"{model_name}_roc.png")

    return acc, f1, model_size, inference_time

if __name__ == "__main__":
    _, val_loader, class_names = get_dataloaders()

    # RESNET
    resnet_model = get_model()
    evaluate_model(
        model=resnet_model,
        model_name="ResNet",
        model_path=f"{MODEL_DIR}/resnet_final.pth",
        val_loader=val_loader,
        class_names=class_names
    )

    # DARTS
    darts_model = DARTSNetwork()
    print(f"\n🧬 DARTS Genotype: {genotype}")
    evaluate_model(
        model=darts_model,
        model_name="DARTS",
        model_path=f"{MODEL_DIR}/darts_final.pth",
        val_loader=val_loader,
        class_names=class_names
    )
