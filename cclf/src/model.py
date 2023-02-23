import torchvision
import torch
from torchvision import transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
auto_transforms = weights.transforms()
model = torchvision.models.efficientnet_b0(weights=weights).to(DEVICE)


custom_transforms = transforms.Compose(
    [
        transforms.TrivialAugmentWide(num_magnitude_bins=31),
    ]
)

all_transforms = transforms.Compose([custom_transforms, auto_transforms])

for param in model.features.parameters():
    param.requires_grad = False

model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True),
    torch.nn.Linear(in_features=1280, out_features=1, bias=True),
).to(DEVICE)
