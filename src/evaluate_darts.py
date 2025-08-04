import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import time
import numpy as np
import os

from nas_darts_model import DARTSNetwork

# ========== Config ==========
DATA_DIR = 'data/raw/cervicalCancer'
MODEL_PATH = 'outputs/models/darts_final.pth'
RESULTS_DIR = 'outputs/results/'
IMG_SIZE = 64
BATCH_SIZE = 64
NUM_CLASSES = 5

# ========== Prepare Dataset ==========
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
class_names = dataset.classes

val_size = int(0.2 * len(dataset))
train_size = len(dataset) - val_size
_, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

# ========== Load Model ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DARTSNetwork()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# ========== Evaluation ==========
all_preds = []
all_labels = []
total_time = 0

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)

        start = time.time()
        outputs = model(images)
        total_time += time.time() - start

        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ========== Metrics ==========
accuracy = np.mean(np.array(all_preds) == np.array(all_labels)) * 100
f1_macro = f1_score(all_labels, all_preds, average='macro')
f1_weighted = f1_score(all_labels, all_preds, average='weighted')

# Model Size
model_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)

# Inference Time per Image
inference_time = (total_time / len(val_loader.dataset)) * 1000  # in ms

print(f"\n📊 DARTS Evaluation Results:")
print(f"Accuracy        : {accuracy:.2f}%")
print(f"F1-score (Macro): {f1_macro:.4f}")
print(f"F1-score (Weighted): {f1_weighted:.4f}")
print(f"Model Size      : {model_size:.2f} MB")
print(f"Inference Time  : {inference_time:.2f} ms/image")

# ========== Confusion Matrix ==========
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix - DARTS")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "darts_confusion_matrix.png"))
plt.close()

# ========== Classification Report ==========
report = classification_report(all_labels, all_preds, target_names=class_names)
with open(os.path.join(RESULTS_DIR, "darts_classification_report.txt"), "w") as f:
    f.write(report)

# ========== ROC Curve ==========
# Convert to one-hot for ROC
all_labels_oh = np.eye(NUM_CLASSES)[all_labels]
all_outputs = []

with torch.no_grad():
    for images, _ in val_loader:
        images = images.to(device)
        outputs = model(images)
        all_outputs.append(outputs.cpu().numpy())

all_outputs = np.concatenate(all_outputs, axis=0)
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(NUM_CLASSES):
    fpr[i], tpr[i], _ = roc_curve(all_labels_oh[:, i], all_outputs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(8, 6))
for i in range(NUM_CLASSES):
    plt.plot(fpr[i], tpr[i], label=f"{class_names[i]} (AUC = {roc_auc[i]:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve - DARTS")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "darts_roc_curve.png"))
plt.close()
