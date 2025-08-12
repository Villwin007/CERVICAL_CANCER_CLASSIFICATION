import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from nas_darts_model import DARTSNetwork

def load_labels():
    with open("labels.txt", "r") as f:
        return [line.strip() for line in f.readlines()]

def load_model(model_path="darts_final.pth", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = load_labels()
    model = DARTSNetwork(num_classes=len(labels))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

def predict(model, device, file):
    image = Image.open(file).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        top_prob, top_class = probs.max(dim=1)

    labels = load_labels()
    return {
        "class": labels[top_class.item()],
        "confidence": float(top_prob.item())
    }
