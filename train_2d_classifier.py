import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

CLASSES = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']

# 1. Lightweight 2D CNN Multi-Class Classifier Architecture
class LightweightClassifier2D(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 128x128
            
            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 64x64
            
            # Block 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 32x32
        )
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        logits = self.classifier(x)
        return logits

# 2. PyTorch Dataset for 4-Class Classification
class MRIClassificationDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.samples = []
        
        for c_idx, cname in enumerate(CLASSES):
            cdir = os.path.join(root_dir, cname)
            if os.path.exists(cdir):
                for fname in sorted(os.listdir(cdir)):
                    if fname.endswith('.png'):
                        self.samples.append((os.path.join(cdir, fname), c_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('L').resize((256, 256), Image.Resampling.BILINEAR)
        
        # Min-Max Normalization (0.0 to 1.0)
        img_arr = np.array(img, dtype=np.float32)
        min_val, max_val = img_arr.min(), img_arr.max()
        if max_val > min_val:
            normalized_img = (img_arr - min_val) / (max_val - min_val + 1e-8)
        else:
            normalized_img = img_arr / 255.0
            
        tensor = torch.from_numpy(normalized_img).unsqueeze(0) # (1, 256, 256)
        return tensor, torch.tensor(label, dtype=torch.long)

def train_classifier(epochs: int = 20, batch_size: int = 8, lr: float = 1e-3):
    os.makedirs("models", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Classifier on device: {device}")
    
    train_dataset = MRIClassificationDataset("classification_dataset/train")
    val_dataset = MRIClassificationDataset("classification_dataset/val")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = LightweightClassifier2D(num_classes=len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    best_val_loss = float('inf')
    model_save_path = os.path.join("models", "brain_tumor_classifier_2d.pth")
    
    print(f"Beginning {epochs}-epoch 2D Multi-Class Classifier training on {len(train_dataset)} samples...")
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        correct_train = 0
        
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct_train += (preds == labels).sum().item()
            
        train_loss /= len(train_dataset)
        train_acc = (correct_train / len(train_dataset)) * 100.0
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(dim=1)
                correct_val += (preds == labels).sum().item()
                
        val_loss /= len(val_dataset)
        val_acc = (correct_val / len(val_dataset)) * 100.0
        
        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} (Acc: {train_acc:.1f}%) | Val Loss: {val_loss:.4f} (Acc: {val_acc:.1f}%)")
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'state_dict': model.state_dict(),
                'classes': CLASSES
            }, model_save_path)
            
    total_time = time.time() - start_time
    print(f"Classifier training completed in {total_time:.2f} seconds.")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Saved trained PyTorch Classifier checkpoint to: {model_save_path}")

if __name__ == "__main__":
    train_classifier()
