from torch.utils.data import DataLoader
import torch
import os
from torchvision import datasets, transforms
from utils import set_seed
from config import TRAIN_DIR, TEST_DIR, BATCH_SIZE, SEED
from model import auto_transforms, all_transforms
from utils import set_seed, loss_curves, save_model
from config import SEED, EPOCHS, LOSS_FN, OPTIMISER
from model import model
from train_utils import train

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = os.cpu_count()


train_dataloader = DataLoader(
    dataset=datasets.ImageFolder(
        root=TRAIN_DIR, transform=all_transforms, target_transform=None
    ),
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=True,
)
test_dataloader = DataLoader(
    dataset=datasets.ImageFolder(root=TEST_DIR, transform=auto_transforms),
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=False,
)

if __name__ == "__main__":
    set_seed(SEED)
    results, model = train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=OPTIMISER,
        loss_fn=LOSS_FN,
        device=DEVICE,
        epochs=EPOCHS,
    )

    loss_curves(results, save_img=True, out_path="../train_logs/loss_curves.png")

    save_model(
        model=model, target_dir="../models", model_name="ccfl_efficient_net_0.pth"
    )
