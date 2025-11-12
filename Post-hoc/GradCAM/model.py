import torch
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
from torchvision.io import decode_image
import json
class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT,progress=False).eval()
        self.transforms = ResNet18_Weights.DEFAULT.transforms(antialias=True)
    def forward(self, x:torch.Tensor):
        with torch.no_grad():
            x = self.transforms(x)
            y = self.model(x)
            return y.argmax(dim=1)
dog1 = decode_image('dog1.jpg')
dog2 = decode_image('dog2.jpg')
device = "cuda" if torch.cuda.is_available() else "cpu"

batch = torch.stack([dog1, dog2]).to(device)
predictor = Predictor().to(device)
res = predictor(batch)
with open('imagenet_class_index.json') as labels_file:
    labels = json.load(labels_file)
for i, pred in enumerate(res):
    print(f"Dog {i} predicted as {labels[str(pred.item())]}")