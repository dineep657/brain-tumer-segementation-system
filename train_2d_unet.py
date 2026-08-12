import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

# 1. Lightweight 2D UNet Architecture with Stable GroupNorm
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # GroupNorm(4, out_ch) provides evaluation stability independent of batch size
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(4, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(4, out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class LightweightUNet2D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.inc = DoubleConv(in_channels, 16)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(16, 32))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv_up1 = DoubleConv(128, 64)
        
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv_up2 = DoubleConv(64, 32)
        
        self.up3 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.conv_up3 = DoubleConv(32, 16)
        
        self.outc = nn.Conv2d(16, out_channels, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up1(x)
        
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up2(x)
        
        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up3(x)
        
        logits = self.outc(x)
        return logits

# 2. PyTorch Dataset with Standardized Min-Max Intensity Scaling
class MRISegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.filename_list = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.filename_list)

    def __getitem__(self, idx):
        fname = self.filename_list[idx]
        img_path = os.path.join(self.img_dir, fname)
        mask_path = os.path.join(self.mask_dir, fname)
        
        img = Image.open(img_path).convert('L').resize((256, 256), Image.Resampling.BILINEAR)
        mask = Image.open(mask_path).convert('L').resize((256, 256), Image.Resampling.NEAREST)
        
        # Min-Max Intensity Scaling (0.0 to 1.0)
        img_arr = np.array(img, dtype=np.float32)
        min_val, max_val = img_arr.min(), img_arr.max()
        if max_val > min_val:
            normalized_img = (img_arr - min_val) / (max_val - min_val + 1e-8)
        else:
            normalized_img = img_arr / 255.0
            
        mask_arr = (np.array(mask, dtype=np.float32) > 128.0).astype(np.float32)
        
        img_tensor = torch.from_numpy(normalized_img).unsqueeze(0)   # (1, 256, 256)
        mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0) # (1, 256, 256)
        return img_tensor, mask_tensor

# 3. Dice + BCE Loss Function
class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets, smooth=1.0):
        bce_loss = self.bce(inputs, targets)
        probs = torch.sigmoid(inputs)
        
        inputs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (inputs_flat * targets_flat).sum()
        dice_score = (2.0 * intersection + smooth) / (inputs_flat.sum() + targets_flat.sum() + smooth)
        dice_loss = 1.0 - dice_score
        
        return bce_loss + dice_loss

def train_model(epochs: int = 25, batch_size: int = 8, lr: float = 1e-3):
    os.makedirs("models", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    train_dataset = MRISegmentationDataset("dataset/train/images", "dataset/train/masks")
    val_dataset = MRISegmentationDataset("dataset/val/images", "dataset/val/masks")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = LightweightUNet2D(in_channels=1, out_channels=1).to(device)
    criterion = DiceBCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    best_val_loss = float('inf')
    model_save_path = os.path.join("models", "brain_tumor_unet_2d.pth")
    
    print(f"Beginning {epochs}-epoch 2D UNet training loop on {len(train_dataset)} training samples...")
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            
        train_loss /= len(train_dataset)
        
        # Validation Loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * imgs.size(0)
                
        val_loss /= len(val_dataset)
        
        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds.")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Saved trained PyTorch checkpoint to: {model_save_path}")

if __name__ == "__main__":
    train_model()
