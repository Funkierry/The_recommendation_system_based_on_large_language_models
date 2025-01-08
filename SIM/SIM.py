import os
import torch
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import resnet50, ResNet50_Weights
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import warnings
import matplotlib.pyplot as plt
import logging
import csv

# Configure logging
RESULTS_DIR = r'C:\Users\User\Desktop\FYP\results'
os.makedirs(RESULTS_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(RESULTS_DIR, 'training.log'),
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)

# Ignore FutureWarning and other warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Paths
BASE_DIR = r'C:\Users\User\Desktop\FYP\dataset'  # Base directory of the dataset
TRAIN_CSV = os.path.join(BASE_DIR, 'train_updated.csv')
VAL_CSV = os.path.join(BASE_DIR, 'val_updated.csv')  # Initially, will be merged
TEST_CSV = os.path.join(BASE_DIR, 'test_updated.csv')
IMG_DIR = BASE_DIR  # Unified img_dir set to BASE_DIR

# Data transformations with additional data augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # Data augmentation
    transforms.RandomRotation(10),      # Data augmentation
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Data augmentation
    transforms.RandomCrop(224, padding=4),  # Additional augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Read data with specified encoding to fix UnicodeDecodeError
def read_csv_with_encoding(file_path):
    try:
        return pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        logging.warning(f"UTF-8 decoding failed for {file_path}. Trying 'latin1' encoding.")
        return pd.read_csv(file_path, encoding='latin1')

train_df = read_csv_with_encoding(TRAIN_CSV)
val_df = read_csv_with_encoding(VAL_CSV)
test_df = read_csv_with_encoding(TEST_CSV)

# Combine training and test sets
full_train_df = pd.concat([train_df, test_df], ignore_index=True)

# Combine all labels to ensure LabelEncoder covers all classes
all_labels = pd.concat([full_train_df['categoryName'], val_df['categoryName']]).unique()

# Use LabelEncoder to encode all category names to integers
label_encoder = LabelEncoder()
label_encoder.fit(all_labels)

# Map labels
full_train_df['label'] = label_encoder.transform(full_train_df['categoryName'])
val_df['label'] = label_encoder.transform(val_df['categoryName'])

# Get number of classes
num_classes = len(label_encoder.classes_)
logging.info(f"Number of classes: {num_classes}")
print(f"Number of classes: {num_classes}")

# Identify classes with only one sample
class_counts = full_train_df['label'].value_counts()
singleton_classes = class_counts[class_counts == 1].index.tolist()
print(f"Singleton classes (only 1 sample): {label_encoder.inverse_transform(singleton_classes)}")
logging.info(f"Singleton classes (only 1 sample): {label_encoder.inverse_transform(singleton_classes)}")

# Separate singleton samples
singleton_df = full_train_df[full_train_df['label'].isin(singleton_classes)]
non_singleton_df = full_train_df[~full_train_df['label'].isin(singleton_classes)]

# Optional: Remove singleton classes from validation set if present
# Since validation set was initially separate, but now we're combining train and test,
# singleton classes are part of full_train_df and have been separated.

# Split non-singleton data into training and validation sets
train_df, val_df_new = train_test_split(
    non_singleton_df,
    test_size=0.1,  # 10% for validation
    stratify=non_singleton_df['label'],
    random_state=42
)

# Combine back singleton samples to training set
train_df = pd.concat([train_df, singleton_df], ignore_index=True)

# Now, train_df contains all training samples, including singleton classes
# val_df_new contains validation samples without singleton classes

# Check if image files exist
def check_image_files(dataframe, img_dir, dataset_name):
    missing_files = []
    for img_name in dataframe['imgUrl']:
        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path):
            missing_files.append(img_path)
    if missing_files:
        logging.warning(f"{dataset_name} is missing {len(missing_files)} files:")
        for file in missing_files[:10]:  # Show only first 10 missing files
            logging.warning(file)
        if len(missing_files) > 10:
            logging.warning(f"... Total missing files: {len(missing_files)}")
    else:
        logging.info(f"All image files in {dataset_name} exist.")

check_image_files(train_df, IMG_DIR, "Training Set")
check_image_files(val_df_new, IMG_DIR, "Validation Set")
# No need to check test_df since it's merged

# Custom Dataset class
class ProductDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None, tokenizer=None):
        self.data = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.tokenizer = tokenizer if tokenizer else BertTokenizer.from_pretrained('bert-base-uncased')

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_name = row['imgUrl']  # e.g., 'images/test/test_70240.jpg'
        img_path = os.path.join(self.img_dir, img_name)  # e.g., 'C:\Users\User\Desktop\FYP\dataset\images\test\test_70240.jpg'

        # Check if file exists
        if not os.path.exists(img_path):
            logging.warning(f"Missing file: {img_path}")
            return None

        try:
            # Load image
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            logging.error(f"Error loading image {img_path}: {e}")
            return None

        # Text processing
        title = row['title']
        inputs = self.tokenizer(
            title,
            padding='max_length',
            truncation=True,
            max_length=64,
            return_tensors='pt'
        )

        # Get label and convert to torch.long
        label = torch.tensor(row['label'], dtype=torch.long)

        return {
            'image': image,
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'label': label
        }

    def __len__(self):
        return len(self.data)

# Custom collate_fn to filter out None
def custom_collate(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return {}
    return torch.utils.data.dataloader.default_collate(batch)

# Focal Loss Implementation
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(reduction='none', ignore_index=self.ignore_index)

    def forward(self, inputs, targets):
        logpt = -self.ce(inputs, targets)  # (batch_size)
        pt = torch.exp(logpt)  # (batch_size)
        focal_term = (1 - pt) ** self.gamma  # (batch_size)
        loss = -self.alpha * focal_term * logpt  # (batch_size)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# Attention-based Fusion Model Definition
class FusionModel(nn.Module):
    def __init__(self, num_classes):
        super(FusionModel, self).__init__()
        # Image feature extractor
        self.resnet = nn.Sequential(
            *list(resnet50(weights=ResNet50_Weights.DEFAULT).children())[:-1]
        )
        self.fc1 = nn.Linear(2048, 512)
        self.dropout = nn.Dropout(0.5)  # Increased Dropout rate

        # Text feature extractor
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc2 = nn.Linear(768, 512)

        # Attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)

        # Final classification layer
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, image, input_ids, attention_mask):
        # Image features
        img_features = self.resnet(image).view(image.size(0), -1)  # (batch_size, 2048)
        img_features = torch.relu(self.fc1(img_features))  # (batch_size, 512)
        img_features = self.dropout(img_features)  # Apply Dropout

        # Text features
        text_output = self.bert(input_ids, attention_mask=attention_mask)
        text_features = torch.relu(self.fc2(text_output.last_hidden_state[:, 0, :]))  # [CLS] token's output (batch_size, 512)
        text_features = self.dropout(text_features)  # Apply Dropout

        # Prepare for attention: add a sequence dimension
        img_features = img_features.unsqueeze(1)  # (batch_size, 1, 512)
        text_features = text_features.unsqueeze(1)  # (batch_size, 1, 512)

        # Concatenate image and text features
        combined_features = torch.cat((img_features, text_features), dim=1)  # (batch_size, 2, 512)

        # Apply attention
        attn_output, _ = self.attention(combined_features, combined_features, combined_features)  # (batch_size, 2, 512)

        # Aggregate attended features
        fused_features = attn_output.mean(dim=1)  # (batch_size, 512)

        # Classification
        output = self.fc3(fused_features)  # (batch_size, num_classes)

        return output

# Function to compute evaluation metrics
def compute_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    return accuracy, precision, recall, f1

# Function to save predictions to CSV
def save_predictions(y_true, y_pred, filename):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['True Label', 'Predicted Label'])
        for true, pred in zip(y_true, y_pred):
            writer.writerow([true, pred])
    logging.info(f"Predictions saved to {filename}")

# Function to plot training and validation metrics
def plot_training_history(history, results_dir):
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(14, 12))

    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history['train_accuracy'], 'b-', label='Training Accuracy')
    plt.plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot F1 score
    plt.subplot(2, 2, 3)
    plt.plot(epochs, history['train_f1'], 'b-', label='Training F1 Score')
    plt.plot(epochs, history['val_f1'], 'r-', label='Validation F1 Score')
    plt.title('Training and Validation F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()

    # Plot Precision and Recall
    plt.subplot(2, 2, 4)
    plt.plot(epochs, history['train_precision'], 'b-', label='Training Precision')
    plt.plot(epochs, history['val_precision'], 'r-', label='Validation Precision')
    plt.plot(epochs, history['train_recall'], 'g-', label='Training Recall')
    plt.plot(epochs, history['val_recall'], 'y-', label='Validation Recall')
    plt.title('Training and Validation Precision & Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'training_history.png'))
    plt.show()

# Training and evaluation function
def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'train_precision': [],
        'val_precision': [],
        'train_recall': [],
        'val_recall': [],
        'train_f1': [],
        'val_f1': []
    }
    best_val_f1 = 0
    epochs_no_improve = 0
    early_stop_patience = 5  # Early stopping patience

    # Initialize mixed precision scaler
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        y_true_train, y_pred_train = [], []

        # Training loop with progress bar
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training", ncols=100)
        for batch in train_pbar:
            if not batch:
                continue  # Skip empty batches

            # Move data to device
            image = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label = batch['label'].to(device)

            optimizer.zero_grad()

            # Mixed precision training
            with torch.cuda.amp.autocast():
                # Forward pass
                output = model(image, input_ids, attention_mask)
                loss = criterion(output, label)

            # Backward pass and optimization
            scaler.scale(loss).backward()

            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            y_true_train.extend(label.cpu().numpy())
            y_pred_train.extend(output.argmax(dim=1).cpu().numpy())

            # Update progress bar description
            train_pbar.set_postfix({'Loss': loss.item()})

        # Calculate training loss and metrics
        avg_train_loss = running_loss / len(train_loader)
        train_accuracy, train_precision, train_recall, train_f1 = compute_metrics(y_true_train, y_pred_train)

        # Validation loop with progress bar
        model.eval()
        val_loss = 0.0
        y_true_val, y_pred_val = [], []
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation", ncols=100)
        with torch.no_grad():
            for batch in val_pbar:
                if not batch:
                    continue  # Skip empty batches

                # Move data to device
                image = batch['image'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                label = batch['label'].to(device)

                # Forward pass
                output = model(image, input_ids, attention_mask)
                loss = criterion(output, label)

                val_loss += loss.item()
                y_true_val.extend(label.cpu().numpy())
                y_pred_val.extend(output.argmax(dim=1).cpu().numpy())

                # Update progress bar description
                val_pbar.set_postfix({'Loss': loss.item()})

        # Calculate validation loss and metrics
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy, val_precision, val_recall, val_f1 = compute_metrics(y_true_val, y_pred_val)

        # Scheduler step based on validation loss
        scheduler.step(avg_val_loss)

        # Record history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_accuracy'].append(val_accuracy)
        history['train_precision'].append(train_precision)
        history['val_precision'].append(val_precision)
        history['train_recall'].append(train_recall)
        history['val_recall'].append(val_recall)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)

        # Log current epoch results
        logging.info(f"Epoch {epoch + 1}/{num_epochs} Summary:")
        logging.info(f"Train Loss: {avg_train_loss:.4f} | Accuracy: {train_accuracy:.4f} | Precision: {train_precision:.4f} | Recall: {train_recall:.4f} | F1 Score: {train_f1:.4f}")
        logging.info(f"Val Loss: {avg_val_loss:.4f} | Accuracy: {val_accuracy:.4f} | Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | F1 Score: {val_f1:.4f}\n")

        # Print current epoch results
        print(f"\nEpoch {epoch + 1}/{num_epochs} Summary:")
        print(f"Train Loss: {avg_train_loss:.4f} | Accuracy: {train_accuracy:.4f} | Precision: {train_precision:.4f} | Recall: {train_recall:.4f} | F1 Score: {train_f1:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f} | Accuracy: {val_accuracy:.4f} | Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | F1 Score: {val_f1:.4f}\n")

        # Check for improvement
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, 'best_model.pth'))
            logging.info(f"--> Saved Best Model (Epoch {epoch + 1})\n")
            print(f"--> Saved Best Model (Epoch {epoch + 1})\n")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            logging.info(f"No improvement in F1 Score for {epochs_no_improve} epoch(s).")
            print(f"No improvement in F1 Score for {epochs_no_improve} epoch(s).")

        # Early stopping
        if epochs_no_improve >= early_stop_patience:
            logging.info("Early stopping triggered.")
            print("Early stopping triggered.\n")
            break

    return history

# Testing function
def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    y_true_test, y_pred_test = [], []
    test_pbar = tqdm(test_loader, desc="Testing", ncols=100)
    with torch.no_grad():
        for batch in test_pbar:
            if not batch:
                continue  # Skip empty batches

            # Move data to device
            image = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label = batch['label'].to(device)

            # Forward pass
            output = model(image, input_ids, attention_mask)
            loss = criterion(output, label)

            test_loss += loss.item()
            y_true_test.extend(label.cpu().numpy())
            y_pred_test.extend(output.argmax(dim=1).cpu().numpy())

            # Update progress bar description
            test_pbar.set_postfix({'Loss': loss.item()})

    # Calculate test loss and metrics
    avg_test_loss = test_loss / len(test_loader)
    test_accuracy, test_precision, test_recall, test_f1 = compute_metrics(y_true_test, y_pred_test)

    # Log test results
    logging.info(f"Test Summary:")
    logging.info(f"Test Loss: {avg_test_loss:.4f} | Accuracy: {test_accuracy:.4f} | Precision: {test_precision:.4f} | Recall: {test_recall:.4f} | F1 Score: {test_f1:.4f}\n")

    # Print test results
    print(f"\nTest Summary:")
    print(f"Test Loss: {avg_test_loss:.4f} | Accuracy: {test_accuracy:.4f} | Precision: {test_precision:.4f} | Recall: {test_recall:.4f} | F1 Score: {test_f1:.4f}\n")

    # Save predictions
    save_predictions(y_true_test, y_pred_test, os.path.join(RESULTS_DIR, 'test_predictions.csv'))

    return avg_test_loss, test_accuracy, test_precision, test_recall, test_f1

# Main function
def main():
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    logging.info("Tokenizer initialized.")
    print("Tokenizer initialized.")

    # Create datasets
    train_dataset = ProductDataset(
        dataframe=train_df,
        img_dir=IMG_DIR,  # Unified img_dir set to BASE_DIR
        transform=transform,
        tokenizer=tokenizer
    )
    val_dataset = ProductDataset(
        dataframe=val_df_new,
        img_dir=IMG_DIR,  # Unified img_dir set to BASE_DIR
        transform=transform,
        tokenizer=tokenizer
    )

    logging.info("Datasets created.")
    print("Datasets created.")

    # Create data loaders
    # Adjust num_workers based on the operating system
    num_workers = 4 if os.name != 'nt' else 0  # Windows requires num_workers=0

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,  # Increased batch size to 64
        shuffle=True,
        num_workers=num_workers,
        collate_fn=custom_collate
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,  # Increased batch size to 64
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate
    )

    logging.info("Data loaders created.")
    print("Data loaders created.")

    # Initialize model, loss function, and optimizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = FusionModel(num_classes=num_classes).to(device)
    criterion = FocalLoss(alpha=1, gamma=2, reduction='mean')  # Using Focal Loss
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)  # Using AdamW with weight decay

    # Initialize learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True,
        min_lr=1e-6
    )

    logging.info(f"Model initialized. Using device: {device}")
    print(f"Model initialized. Using device: {device}")

    # Train and evaluate
    num_epochs = 30  
    history = train_and_evaluate(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        device=device
    )

    # Plot training history
    plot_training_history(
        history=history,
        results_dir=RESULTS_DIR
    )

    # Load the best model
    best_model_path = os.path.join(RESULTS_DIR, 'best_model.pth')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        logging.info("Loaded the best model.")
        print("Loaded the best model for testing.\n")
    else:
        logging.warning("Best model not found. Using the current model for testing.")
        print("Best model not found. Using the current model for testing.\n")

    # Test the model
    # Since test_df was merged into training, we can use val_loader as test_loader
    test(
        model=model,
        test_loader=val_loader,  # Using validation loader as test loader
        criterion=criterion,
        device=device
    )

if __name__ == '__main__':
    main()
