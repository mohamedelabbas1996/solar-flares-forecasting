import torch.nn as nn
import wandb
import torch
import pickle
from datasets.dataset import SolarFlaresData
from torch.optim import Adam
from torch.utils.data import DataLoader
from model import CNN


wandb.login()
run = wandb.init(
    # Set the project where this run will be logged
    project="solar-flares-forecasting",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": 0.001,
        "epochs": 2,
    },
)

with open(
    "/Users/mohamedelabbas/solar-flares-forecasting/solar-flares-forecasting/data/sharp/preprocessed/all_sharps.pkl",
    "rb",
) as f:
    df = pickle.load(f)
dataset = SolarFlaresData(df)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)
avg_loss = 0
for epoch in range(2):
    for idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        image = batch[1]
        label = batch[2]

        output = model(image.to(torch.float32))
        loss = criterion(output, label)
        avg_loss += loss
        avg_loss /= idx + 1
        wandb.log({"avg_loss": avg_loss, "loss": loss.item()})
        loss.backward()
        optimizer.step()
