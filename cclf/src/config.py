from pathlib import Path
import torch
from model import model

IMG_PATH = Path("../data/cats/")
TRAIN_DIR = IMG_PATH / "train"
TEST_DIR = IMG_PATH / "test"
SEED = 1809
EPOCHS = 20
BATCH_SIZE = 16
LR = 0.001

LOSS_FN = torch.nn.BCEWithLogitsLoss()
OPTIMISER = torch.optim.Adam(model.parameters(), lr=LR)
