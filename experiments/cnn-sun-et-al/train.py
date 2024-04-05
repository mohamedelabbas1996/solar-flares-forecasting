import sys
sys.path.append("models/cnn")
sys.path.append("datasets/sharp_sun_et_al")
sys.path.append("loss")
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
from dataset import MagnetogramDataset
import torch.nn.functional as F
import datetime
from tqdm import tqdm
from loss import FocalBCELoss

def train(model, optimizer, train_loader, validation_loader, num_epochs, criterion, device, interactive=False):
   
    best_val_tss = float('-inf')
    patience_counter = 0
    patience = 10
    for epoch in range(num_epochs):
        # Use tqdm for the progress bar if debug is True, else iterate normally
        iterable = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") if interactive else train_loader
        all_preds = []
        all_targets = []
        for batch_idx, (data, target) in enumerate(iterable):
            model.train()
            data = data.unsqueeze(1)  # Assuming your model needs this shape
            target = target.unsqueeze(1).float()  # Adjust for your model's expected input
            data, target, model = data.to(device), target.to(device), model.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # Log training loss with wandb
            wandb.log({"train_loss": loss.item()})
            
            loss.backward()
            optimizer.step()
            
            preds = torch.round(torch.sigmoid(output))
            all_preds.extend(preds.view(-1).cpu().detach().numpy())
            all_targets.extend(target.view(-1).cpu().detach().numpy())
            
            if (batch_idx) % 50 == 0:
                accuracy, precision, recall, validation_loss, cm, hss_score, tss_score = validate_model(model, validation_loader, device)
                
                # Log validation metrics with wandb
                wandb.log({
                    "validation_loss": validation_loss,
                    "validation accuracy": accuracy,
                    "validation  tn, fp, fn, tp": str(cm.ravel()),
                    "validation confusion_matrix": cm,
                    "validation precision": precision,
                    "validation recall": recall,
                    "validation TSS": tss_score,
                    "validation HSS": hss_score,
                })
                # Check for improvement in validation TSS
                if tss_score > best_val_tss:
                    best_val_tss = tss_score
                    patience_counter = 0  # Reset patience since we found a better model
                    
                    # Save the best model
                    datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    best_checkpoint_path = f"checkpoints/best_model_checkpoint_{wandb.run.name}.pth"
                    torch.save(model.state_dict(), best_checkpoint_path)
                    wandb.save(best_checkpoint_path)
                else:
                    patience_counter += 1
                    
                    if patience_counter >= patience:
                        print(f"Stopping early at epoch {epoch+1} due to no improvement in validation TSS.")
                        return  # Early stopping trigger
        
        # Save model checkpoint
        datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        checkpoint_path = f"checkpoints/model_checkpoint_epoch_{wandb.run.name}_{epoch}_{datetime_str}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        
        # Save the checkpoint with wandb
        wandb.save(checkpoint_path)

def calculate_hss(true_positives, false_positives, true_negatives, false_negatives):
    E = ((true_positives + false_negatives) * (true_positives + false_positives) + 
         (false_negatives + true_negatives) * (false_positives + true_negatives)) / (true_positives + true_negatives + false_positives + false_negatives)
    HSS = (true_positives + true_negatives - E) / (true_positives + true_negatives + false_positives + false_negatives - E)
    return HSS

def calculate_tss(true_positives, false_positives, true_negatives, false_negatives):
    TSS = (true_positives / (true_positives + false_negatives)) - (false_positives / (false_positives + true_negatives))
    return TSS

def validate_model(model, validation_loader, device, is_test=False):
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
    class_names = ['Quiet', 'Strong']        
    wandb.log({f"{'test' if is_test else 'validation'} confusion_matrix": wandb.plot.confusion_matrix(
    probs=None,
    y_true=all_targets,
    preds=all_preds,
    class_names=class_names)})
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds)
    cm = confusion_matrix(all_targets, all_preds)
    tn, fp, fn, tp = cm.ravel()
    hss_score = calculate_hss(tp, fp, tn, fn)
    tss_score = calculate_tss(tp, fp, tn, fn)

    validation_loss = val_running_loss / len(validation_loader)
    
    return accuracy, precision, recall, validation_loss, cm, hss_score, tss_score
def balance_df(df):
    positive_samples = df[df['label'] == True]
    negative_samples = df[df['label'] == False].sample(n=len(positive_samples), replace=False)
    df = pd.concat([positive_samples, negative_samples]).reset_index()
    return df
def main(args):
    if args.debug:
        print("running the training script")
    # Initialize Weights & Biases
    #wandb.init(project="Solar Flares Forecasting", entity="hack1996man")
    wandb.init(mode="offline")
    # train_df= pd.read_csv("datasets/sharp_sun_et_al/v5/sharp_sun_et_al_df_train_balanced_v5.csv")
    # valid_df= pd.read_csv("datasets/sharp_sun_et_al/v5/sharp_sun_et_al_df_val_balanced_v5.csv")
    # test_df= pd.read_csv("datasets/sharp_sun_et_al/v5/sharp_sun_et_al_df_test_balanced_v5.csv")
    sharp_df = pd.read_csv("datasets/sharp_sun_et_al/sharp_sun_et_al_filtered.csv")


    model = CNN()
    criterion = F.binary_cross_entropy_with_logits
    #criterion = FocalBCELoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    config = dict(
    model_architecture = str(model),
    learning_rate = args.lr,
    optimizer_name = "Adam",
    batch_size =  args.batch_size,
    num_epochs = args.num_epochs,
    loss_function = "F.binary_cross_entropy_with_logits"
)
    wandb.config.update(config)
    wandb.config.update({
        "train_data":"datasets/sharp_sun_et_al/sharp_sun_et_al_filtered.csv",
        
    })
    unique_regions = sharp_df['region_no'].unique()

# Calculate the number of regions for each split
    num_test_regions = int(len(unique_regions) * 0.2)
    num_val_regions = int(len(unique_regions) * 0.2)
    num_train_regions = len(unique_regions) - num_test_regions - num_val_regions

# Randomly select regions for each split
    test_regions = np.random.choice(unique_regions, size=num_test_regions, replace=False)
    

# Split the data based on selected regions
    test_df = sharp_df[sharp_df['region_no'].isin(test_regions)]
    test_df = balance_df(test_df) 
    test_df.to_csv("datasets/sharp_sun_et_al/sharp_sun_et_al_test.csv")
    test_dataset = MagnetogramDataset(test_df, magnetograms_dirs=["data/SHARP/sharp_magnetograms_sun_et_al_decompressed/sharp_magnetograms_sun_et_al_compressed_1","data/SHARP/sharp_data_all_magnetograms"])

    test_loader = DataLoader(
    test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
)
    
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    for i in range(10):
        remaining_regions = np.setdiff1d(unique_regions, test_regions)
        val_regions = np.random.choice(remaining_regions, size=num_val_regions, replace=False)
        train_regions = np.setdiff1d(remaining_regions, val_regions)
    
        train_df = sharp_df[sharp_df['region_no'].isin(val_regions)]
        train_df = balance_df(train_df)
        val_df = sharp_df[sharp_df['region_no'].isin(train_regions)]
        val_df = balance_df(val_df)

        train_dataset = MagnetogramDataset(train_df, magnetograms_dirs=["data/SHARP/sharp_magnetograms_sun_et_al_decompressed/sharp_magnetograms_sun_et_al_compressed_1","data/SHARP/sharp_data_all_magnetograms"])

        train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
        val_dataset = MagnetogramDataset(val_df, magnetograms_dirs=["data/SHARP/sharp_magnetograms_sun_et_al_decompressed/sharp_magnetograms_sun_et_al_compressed_1","data/SHARP/sharp_data_all_magnetograms"])

        valid_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    
        model = CNN()
        optimizer = Adam(model.parameters(), lr=args.lr)
        train(model, optimizer, train_loader, valid_loader, args.num_epochs, criterion, device, interactive=args.debug)
        best_checkpoint = torch.load(f'checkpoints/best_model_checkpoint_{wandb.run.name}.pth')
        model.load_state_dict(best_checkpoint)
        accuracy, precision, recall, validation_loss, cm, hss_score, tss_score = validate_model(model, test_loader, device, is_test=True)
        print (f"Run number {i} ")
        print (f"TSS score {tss_score}")
        print (f"HSS score {hss_score}")
        print (f"Accuracy {accuracy}")
        print (f"Precision {precision}")
        print (f"Recall {recall}")
        wandb.log({
            "Run Number": i,
            "Test accuracy": accuracy,
            "Test tn, fp, fn, tp":str(cm.ravel()),
            "Test confusion_matrix" : cm,
            "Test precision": precision,
            "Test recall": recall,
            "Test TSS": tss_score,
            "Test HSS": hss_score})

    # Log the confusion matrix
   
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Script")

    # Define arguments
    parser.add_argument('--debug', action='store_true', help='Debug enabled')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=2, help='Number of epochs')

    # Parse arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args)
