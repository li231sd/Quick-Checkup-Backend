import torch
from torchvision import transforms
from PIL import Image
from models.abcde.model import ABCDEModel

WEIGHTS_PATH = "models/abcde/abcde_model.pth"
LABELS = ["asymmetry", "border", "color", "diameter", "evolution"]

_model = None
_device = None

def get_model():
    global _model, _device
    if _model is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model = ABCDEModel()
        _model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=_device))
        _model.to(_device)
        _model.eval()
        print(f"[ABCDEModel] Loaded on {_device}")
    return _model, _device


def preprocess(image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)


def run_abcde_model(image: Image.Image) -> dict:
    model, device = get_model()
    tensor = preprocess(image).to(device)
    with torch.no_grad():
        outputs = model(tensor)
    scores = outputs.squeeze().tolist()
    return {label: round(score, 4) for label, score in zip(LABELS, scores)}
