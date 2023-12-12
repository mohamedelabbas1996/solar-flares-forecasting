import torch.nn as nn
import wandb
import torch
import pickle
from datasets.dataset import SolarFlaresData
from torch.optim import AdamW
from torch.utils.data import DataLoader
from model import CNN, ViT
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt


def read_df_from_pickle(path):
    with open(path, "rb") as f:
        df = pickle.load(f)
    return df


def compute_accuracy(predictions, correct):
    return (predictions == correct).sum().item() / predictions.shape[0]


def confusion(prediction, target):
    confusion_vector = prediction / target
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


def compute_tss(prediction, target):
    TP, FP, TN, FN = confusion(prediction, target)
    # print(TP, FP, TN, FN)
    N = TN + FP
    P = TP + FN
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
def region_based_split(dataset_df, train_regions, test_regions):
    train_df = dataset_df[dataset_df.region_no.isin(set(train_regions))]
    test_df = dataset_df[dataset_df.region_no.isin(set(test_regions))]
    return train_df, test_df


lr = 0.001
batch_size = 16
num_epochs = 15
device = "cuda" if torch.cuda.is_available() else "cpu"
df = read_df_from_pickle("data/SHARP/SHARP.pkl")
train_df, test_df = region_based_split(
    df, train_regions=[1, 6206], test_regions=[2, 6327]
)
print(f"regions in train set {train_df.region_no.unique()}")
print(f"regions in validation set{test_df.region_no.unique()}")

train_dataset = SolarFlaresData(train_df.reset_index(), random_undersample=False)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = SolarFlaresData(test_df.reset_index(), random_undersample=False)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True, drop_last=True
)

model = ViT()
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=0.001)
model.train()
train_losses = []
val_losses = []
accuracies = []
tss_scores = []
for epoch in range(1, num_epochs):
    epoch_train_losses = []
    with tqdm(train_loader, unit="batch") as tepoch:
        for _, data, target in tepoch:
            tepoch.set_description(f"Epoch {epoch}")

            data, target = data.to(device), target.to(device)
            logits = model(data)
            predictions = logits.argmax(dim=1, keepdim=True).squeeze()
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            epoch_train_losses.append(loss.item())
            # acc = compute_accuracy(predictions, target)
            # tss = compute_tss(acc)
            tepoch.set_postfix(loss=loss.item())
    with torch.no_grad():
        with tqdm(test_loader, unit="batch") as tepoch:
            batch_correct = []
            batch_tss = []
            batch_valid_losses = []
            all_predictions = []
            all_targets = []
            for _, data, target in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                data, target = data.to(device), target.to(device)
                logits = model(data)
                predictions = logits.argmax(dim=1, keepdim=True).squeeze()
                all_predictions.append(predictions)
                all_targets.append(target)
                loss = criterion(logits, target)
                batch_valid_losses.append(loss.item())
                correct = (predictions == target).sum().item()
                batch_acc = compute_accuracy(predictions, target)
                batch_correct.append(correct)

                tepoch.set_postfix(
                    valid_loss=loss.item(),
                )
    train_losses.append(np.mean(epoch_train_losses))
    val_losses.append(np.mean(batch_valid_losses))
    accuracies.append(np.sum(batch_correct) / len(test_dataset))
    all_predictions = torch.cat(all_predictions, axis=0)
    all_targets = torch.cat(all_targets, axis=0)
    # print(all_targets.shape, all_predictions.shape)
    tss_scores.append(compute_tss(all_predictions, all_targets))
plt.figure(figsize=(12, 4))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(range(len(accuracies)), accuracies, label="Accuracy", color="blue")
plt.plot(range(len(tss_scores)), tss_scores, label="TSS Score", color="green")
plt.title("Accuracy and TSS Score")
plt.xlabel("Epoch")
plt.ylabel("Metric Value")
plt.legend()

# Plot Training and Validation Loss
plt.subplot(1, 2, 2)
plt.plot(range(len(train_losses)), train_losses, label="Training Loss", color="red")
plt.plot(range(len(val_losses)), val_losses, label="Validation Loss", color="purple")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss Value")
plt.legend()
plt.tight_layout()
plt.show()

print("Experiment summary")
print(f"Number of epochs {num_epochs}")
print(f"batch size {batch_size}")
print(
    f"train set length {len(train_df)} with {len(train_df[train_df.label == 1])} positive examples and {len(train_df[train_df.label == 0])} negative examples"
)

print(
    f"test set length {len(test_df)} with {len(test_df[test_df.label == 1])} positive examples and {len(test_df[test_df.label == 0])} negative examples"
)
print(f"max accuracy {max(accuracies)}")
print(f"max tss score {max(tss_scores)}")
