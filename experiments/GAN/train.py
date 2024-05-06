import torch
import sys
sys.path.append("models/GAN")
sys.path.append("datasets/sharp_sun_et_al")
from model import Discriminator, Generator
import torchvision.utils as vutils
import wandb
import pandas as pd 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import MagnetogramDataset
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import torch.nn.functional as F

def train(netD, netG,optimizerD, optimizerG, num_epochs, dataloader, criterion, device):
    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    real_label = 1.
    fake_label = 0.
    image_size = 64

    fixed_noise = torch.randn(64, 100, 1, 1, device=device)
    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, (data, target) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data.to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, 100, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1
    


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def balance_df(df):
    positive_samples = df[df['label'] == True]
    negative_samples = df[df['label'] == False].sample(n=len(positive_samples), replace=False)
    df = pd.concat([positive_samples, negative_samples]).reset_index()
    return df


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
            target = target.unsqueeze(1).float()  
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

def main():
    # Learning rate for optimizers
    lr = 0.0002
    # Beta1 hyperparameter for Adam optimizers
    beta1 = 0.5
    batch_size = 64
    num_epochs = 3
    img_size = 64

    sharp_df = pd.read_csv("datasets/sharp_sun_et_al/sharp_sun_et_al_filtered.csv")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    netG = Generator(1).to(device)
    netG.apply(weights_init)
    netD = Discriminator(1).to(device)
    netD.apply(weights_init)
    criterion = nn.BCELoss()
    

    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999)) 


    wandb.init(mode="offline")
    config = dict(
    model_architecture = str(optimizerD) + "\n"+str(optimizerG),
    learning_rate = lr,
    optimizer_name = f"optimizerD = Adam(netD.parameters(), lr={lr}, betas=({beta1}, 0.999))\n optimizerG = Adam(netG.parameters(), lr={lr}, betas=({beta1}, 0.999)) ",
    batch_size =  batch_size,
    num_epochs = num_epochs,
    loss_function = "nn.BCELoss()"
)
    wandb.config.update(config)
    


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
    test_dataset = MagnetogramDataset(test_df, magnetograms_dirs=["data/SHARP/sharp_magnetograms_sun_et_al_decompressed/sharp_magnetograms_sun_et_al_compressed_1","data/SHARP/sharp_data_all_magnetograms"], resize=64)

    test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True, drop_last=True
)
    remaining_regions = np.setdiff1d(unique_regions, test_regions)
    val_regions = np.random.choice(remaining_regions, size=num_val_regions, replace=False)
    train_regions = np.setdiff1d(remaining_regions, val_regions)
    
    train_df = sharp_df[sharp_df['region_no'].isin(val_regions)]
    train_df = balance_df(train_df)
    val_df = sharp_df[sharp_df['region_no'].isin(train_regions)]
    val_df = balance_df(val_df)

    train_df = train_df[train_df['label']== True]
    train_dataset = MagnetogramDataset(train_df, magnetograms_dirs=["data/SHARP/sharp_magnetograms_sun_et_al_decompressed/sharp_magnetograms_sun_et_al_compressed_1","data/SHARP/sharp_data_all_magnetograms"], resize=64)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    # val_dataset = MagnetogramDataset(val_df, magnetograms_dirs=["data/SHARP/sharp_magnetograms_sun_et_al_decompressed/sharp_magnetograms_sun_et_al_compressed_1","data/SHARP/sharp_data_all_magnetograms"])

    # valid_loader = DataLoader(
    # val_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    # )

    train(netD, netG, optimizerD, optimizerG, num_epochs, train_loader, criterion, device)
    accuracy, precision, recall, validation_loss, cm, hss_score, tss_score = validate_model(netD, test_loader, device, is_test=True)
    print (f"TSS score {tss_score}")
    print (f"HSS score {hss_score}")
    print (f"Accuracy {accuracy}")
    print (f"Precision {precision}")
    print (f"Recall {recall}")
    wandb.log({
        
            "Test accuracy": accuracy,
            "Test tn, fp, fn, tp":str(cm.ravel()),
            "Test confusion_matrix" : cm,
            "Test precision": precision,
            "Test recall": recall,
            "Test TSS": tss_score,
            "Test HSS": hss_score})
if __name__ == "__main__":
    main()