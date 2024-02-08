import os
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from torchvision import transforms
from PIL import Image
import pydicom
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score


class CustomDataset(Dataset):
    def __init__(self, csv_file, data_dir, transform=None):
        self.data = pd.read_csv(csv_file).drop_duplicates(subset='patientId', keep='first')
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, f"{self.data['patientId'].iloc[idx]}.dcm")
        image = self.load_dicom_image(img_name)
        label = torch.tensor(self.data['Target'].iloc[idx], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label

    def load_dicom_image(self, path):
        dcm = pydicom.dcmread(path)
        image = dcm.pixel_array.astype(float)
        # You might need to adjust preprocessing steps based on your requirements

        return Image.fromarray(image)


# Example usage
data_dir = "./data/stage_2_train_images/"
csv_file = "./data/stage_2_train_labels.csv"

# Define transformations if needed (e.g., resizing, normalization, etc.)
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # ResNet50 input size
    transforms.ToTensor(),
])

# Create custom dataset
custom_dataset = CustomDataset(csv_file=csv_file, data_dir=data_dir, transform=transform)

# Split the dataset into training and validation sets
train_size = int(0.9 * len(custom_dataset))
val_size = len(custom_dataset) - train_size
train_dataset, val_dataset = random_split(custom_dataset, [train_size, val_size])

# Create DataLoader for training and validation
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8)

torch.cuda.empty_cache()
device = "cuda" if torch.cuda.is_available() else "cpu"

# ResNet-50
model = torchvision.models.resnet34(weights="IMAGENET1K_V1")
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
num_classes = 1
model.fc = nn.Linear(model.fc.in_features, num_classes)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

torch.cuda.empty_cache()


def calculate_f1_score(predictions, targets, threshold=0.5):
    predictions = (predictions > threshold).float()

    # Convert predictions and targets to numpy arrays
    predictions_np = predictions.cpu().numpy()
    targets_np = targets.cpu().numpy()

    # Calculate precision, recall, and F1 score
    precision = precision_score(targets_np, predictions_np, zero_division=0)
    recall = recall_score(targets_np, predictions_np, zero_division=0)
    f1 = f1_score(targets_np, predictions_np, zero_division=0)

    return precision, recall, f1


num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0.0
    precision = 0.0
    recall = 0.0
    f1 = 0.0

    # Wrap your train_dataloader with tqdm
    for inputs, labels in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs).flatten()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Log relevant information during training
        pr, rec, fone = calculate_f1_score(outputs, labels)
        precision += pr
        recall += rec
        f1 += fone
        total_train_loss += loss

    average_train_loss = total_train_loss / len(train_dataloader)
    avg_prec_train = precision / len(train_dataloader)
    avg_rec_train = recall / len(train_dataloader)
    avg_f1_train = f1 / len(train_dataloader)

    # Validation loop
    model.eval()
    total_vall_loss = 0.0
    precision = 0.0
    recall = 0.0
    f1 = 0.0
    with torch.no_grad():
        for inputs, labels in val_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs).flatten()
            loss = criterion(outputs, labels)

            # Log relevant information during training
            pr, rec, fone = calculate_f1_score(outputs, labels)
            precision += pr
            recall += rec
            fone += f1
            total_vall_loss += loss.item()

    average_val_loss = total_vall_loss / len(val_dataloader)
    avg_prec_val = precision / len(val_dataloader)
    avg_rec_val = recall / len(val_dataloader)
    avg_f1_val = fone / len(val_dataloader)

    # Log information using logging
    print(
        f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {average_train_loss}, Train prec: {avg_prec_val}, \
        Train rec: {avg_rec_train}, Train f1: {avg_f1_train}, \
        Val Loss: {average_val_loss}, Val prec: {avg_prec_val}, \
        Val rec: {avg_rec_val}, Val f1: {avg_f1_val}"
    )

    # Save model state
    torch.save(model.state_dict(), f"./models/model{epoch + 1}.pth")

print("Training complete.")
