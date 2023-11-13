import torch.nn as nn
import wandb
import torch
import pickle
from datasets.dataset import SolarFlaresData
from torch.optim import Adam
from torch.utils.data import DataLoader
from model import CNN
from tqdm import tqdm
import torch


def read_df_from_pickle(path):
    with open(path, "rb") as f:
        df = pickle.load(f)
    return df


def confusion(prediction, correct):
    confusion_vector = prediction / correct
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float("inf")).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    return true_positives, false_positives, true_negatives, false_negatives


def compute_accuracy(predictions, correct):
    return (predictions == correct).sum().item() / predictions.shape[0]


def compute_tss(predicitons, correct):
    print(predicitons.shape, correct.shape)
    TP, FP, TN, FN = confusion(predicitons, correct)
    print(TP, FP, TN, FN)
    P = TP + FP
    N = TN + FN
    return TP / P - FP / N


# wandb.login()
# run = wandb.init(
#     # Set the project where this run will be logged
#     project="solar-flares-forecasting",
#     # Track hyperparameters and run metadata
#     config={
#         "learning_rate": 0.001,
#         "epochs": 2,
#     },
# )
lr = 0.001
batch_size = 64
device = "cuda" if torch.cuda.is_available() else "cpu"
df = read_df_from_pickle("data/SHARP/SHARP.pkl")
train_dataset = SolarFlaresData(df)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = SolarFlaresData(read_df_from_pickle("data/SMARP/SMARP.pkl"))
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)
model.train()
for epoch in range(1, 5):
    with tqdm(train_loader, unit="batch") as tepoch:
        for _, data, target in tepoch:
            tepoch.set_description(f"Epoch {epoch}")

            data, target = data.to(device), target.to(device)
            target = target
            logits = model(data)
            predictions = logits.argmax(dim=1, keepdim=True).squeeze()
            print(target, predictions)
            # quit()
            # print(logits.shape, target.shape)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            tepoch.set_postfix(
                loss=loss.item(),
                accuracy=100.0 * compute_accuracy(predictions, target),
                tss=compute_tss(predictions, target),
            )
    with torch.no_grad():
        with tqdm(test_loader, unit="batch") as tepoch:
            for _, data, target in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                data, target = data.to(device), target.to(device)
                logits = model(data)
                predictions = logits.argmax(dim=1, keepdim=True).squeeze()
                print(target, predictions)
                # quit()
                # print(logits.shape, target.shape)
                loss = criterion(logits, target)

                tepoch.set_postfix(
                    valid_loss=loss.item(),
                    valid_accuracy=100.0 * compute_accuracy(predictions, target),
                    valid_tss=compute_tss(predictions, target),
                )
