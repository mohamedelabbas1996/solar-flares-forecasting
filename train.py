import wandb
import torch
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import numpy as np
import argparse
import pandas as pd
from model import CNN
from torch.optim import Adam
from torch.utils.data import DataLoader
from datasets.dataset import MagnetogramDataset
import torch.nn.functional as F
import datetime

# Assuming `model` and `optimizer` are defined and initialized according to the hyperparameters
def train(model, optimizer, train_loader, validation_loader, num_epochs,criterion, device):
   
    model.train()
    for epoch in range(num_epochs):
        all_preds = []
        all_targets = []
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.unsqueeze(1)  # Assuming your model needs this shape
            target = target.unsqueeze(1).float()  # Adjust for your model's expected input
            data, target, model = data.to(device), target.to(device), model.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            wandb.log({"train_loss": loss.item()})
            loss.backward()
            optimizer.step()
            
            preds = torch.round(torch.sigmoid(output))
            all_preds.extend(preds.view(-1).cpu().detach().numpy())
            all_targets.extend(target.view(-1).cpu().detach().numpy())
            
            if (batch_idx+1) % 100== 0:
                accuracy, precision, recall, validation_loss, cm, hss_score, tss_score = validate_model(model, validation_loader, device)
                wandb.log({"validation_loss": validation_loss})
                 # Log metrics
                wandb.log({
            "accuracy": accuracy,
            " tn, fp, fn, tp":str(cm.ravel()),
            "confusion_matrix" : cm,
            "precision": precision,
            "recall": recall,
            "TSS": tss_score,
            "HSS": hss_score,
        })
        
        
       
        
        # Save model checkpoint
        datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")        
        checkpoint_path = f"checkpoints/model_checkpoint_epoch_{wandb.run.name}_{epoch}_{datetime_str}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        wandb.save(checkpoint_path)
    
    # Log the final model as an artifact
    model_artifact = wandb.Artifact('model_weights', type='model')
    model_artifact.add_file(f'checkpoints/model_checkpoint_epoch_{wandb.run.name}_{epoch}_{datetime_str}.pth')
    wandb.log_artifact(model_artifact)


def calculate_hss(true_positives, false_positives, true_negatives, false_negatives):
    E = ((true_positives + false_negatives) * (true_positives + false_positives) + 
         (false_negatives + true_negatives) * (false_positives + true_negatives)) / (true_positives + true_negatives + false_positives + false_negatives)
    HSS = (true_positives + true_negatives - E) / (true_positives + true_negatives + false_positives + false_negatives - E)
    return HSS

def calculate_tss(true_positives, false_positives, true_negatives, false_negatives):
    TSS = (true_positives / (true_positives + false_negatives)) - (false_positives / (false_positives + true_negatives))
    return TSS

def validate_model(model, validation_loader, device):
    model.eval()
    val_running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in validation_loader:
            data = data.unsqueeze(1)  # Assuming your model needs this shape
            target = target.unsqueeze(1).float()  # Adjust for your model's expected input
            data, target, model = data.to(device), target.to(device), model.to(device)
            output = model(data)
            loss = F.binary_cross_entropy_with_logits(output, target)
            val_running_loss += loss.item()

            preds = torch.round(torch.sigmoid(output))
            all_preds.extend(preds.view(-1).cpu().numpy())
            all_targets.extend(target.view(-1).cpu().numpy())

    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds)
    cm = confusion_matrix(all_targets, all_preds)
    tn, fp, fn, tp = cm.ravel()
    hss_score = calculate_hss(tp, fp, tn, fn)
    tss_score = calculate_tss(tp, fp, tn, fn)

    validation_loss = val_running_loss / len(validation_loader)
    
    return accuracy, precision, recall, validation_loss, cm, hss_score, tss_score

def main(args):
    # Initialize Weights & Biases
    #wandb.init(project="Solar Flares Forecasting", entity="hack1996man")
    wandb.init(mode="offline")
    

    # active region based split 
    data_df = pd.read_csv("data/SHARP/sharp_dataset_20000.csv")
   
    unique_regions = data_df['harp_no'].unique()

# Shuffle the unique regions to ensure random splits
    np.random.seed(42)
    np.random.shuffle(unique_regions)

# Calculate split sizes
    train_size = int(len(unique_regions) * 0.7)
    valid_size = int(len(unique_regions) * 0.15)

# Split the unique regions
    train_regions = unique_regions[:train_size]
    valid_regions = unique_regions[train_size:train_size + valid_size]
    test_regions = unique_regions[train_size + valid_size:]
    
    train_mask = data_df['harp_no'].isin(train_regions)

    valid_mask = data_df['harp_no'].isin(valid_regions)
    test_mask = data_df['harp_no'].isin(test_regions)

    # Segment the DataFrame
    train_df = data_df[train_mask]
    valid_df = data_df[valid_mask]
    test_df = data_df[test_mask]

    # save splitted data
    train_df.to_csv("data/SHARP/train_df.csv")
    valid_df.to_csv("data/SHARP/valid_df.csv")
    test_df.to_csv("data/SHARP/test_df.csv")



    model = CNN()
    criterion = F.binary_cross_entropy_with_logits
    optimizer = Adam(model.parameters(), lr=args.lr)

    config = dict(
    model_architecture = str(model),
    learning_rate = args.lr,
    optimizer_name = "Adam",
    batch_size =  args.batch_size,
    num_epochs = args.num_epochs,
    train_dataset_file_name = "data/SHARP/train_df.csv",
    validation_dataset_file_name ="data/SHARP/test_df.csv",
    loss_function = "F.binary_cross_entropy_with_logits"
)
    wandb.config.update(config)
    wandb.config.update({
        "train_data":"data/SHARP/train_df.csv",
        "validation_data":"data/SHARP/valid_df.csv",
        "test_data":"data/SHARP/test_df.csv"
    })
    # initilize model, optimizer, loss, datasets, train,test loaders
    train_dataset = MagnetogramDataset(train_df, magnetograms_dir="data/SHARP/magnetograms/content/drive/MyDrive/sharp_data")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    valid_dataset = MagnetogramDataset(valid_df, magnetograms_dir="data/SHARP/magnetograms/content/drive/MyDrive/sharp_data")
    valid_loader = DataLoader(
    valid_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
)
    test_dataset = MagnetogramDataset(test_df, magnetograms_dir="data/SHARP/magnetograms/content/drive/MyDrive/sharp_data")

    test_loader = DataLoader(
    test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
)
    
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    train(model, optimizer, train_loader, valid_loader, args.num_epochs, criterion, device)
    accuracy, precision, recall, validation_loss, cm, hss_score, tss_score = validate_model(model, test_loader, device)

    wandb.log({
            "Test accuracy": accuracy,
            "Test tn, fp, fn, tp":str(cm.ravel()),
            "Test confusion_matrix" : cm,
            "Test precision": precision,
            "Test recall": recall,
            "Test TSS": tss_score,
            "Test HSS": hss_score})
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Script")

    # Define arguments
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=2, help='Number of epochs')

    # Parse arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args)
