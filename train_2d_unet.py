import os
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

torch.set_num_threads(os.cpu_count() or 4)

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
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

class DiceBCEWithLogitsLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (probs_flat * targets_flat).sum()
        dice_score = (2.0 * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        dice_loss = 1.0 - dice_score
        
        return bce_loss + dice_loss

class RealMRISegmentationDataset(Dataset):
    def __init__(self, root_dir, augment=False, positive_only=True):
        self.img_dir = os.path.join(root_dir, "images")
        self.mask_dir = os.path.join(root_dir, "masks")
        self.augment = augment
        all_fnames = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.png')])
        
        self.filenames = []
        for fname in all_fnames:
            mpath = os.path.join(self.mask_dir, fname)
            mask_arr = np.array(Image.open(mpath))
            if not positive_only or mask_arr.max() > 0:
                self.filenames.append(fname)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        ipath = os.path.join(self.img_dir, fname)
        mpath = os.path.join(self.mask_dir, fname)
        
        img = Image.open(ipath).convert('L').resize((128, 128), Image.Resampling.BILINEAR)
        mask = Image.open(mpath).convert('L').resize((128, 128), Image.Resampling.NEAREST)
        
        if self.augment and random.random() > 0.5:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            
        img_arr = np.array(img, dtype=np.float32)
        min_v, max_v = img_arr.min(), img_arr.max()
        norm_img = (img_arr - min_v) / (max_v - min_v + 1e-8) if max_v > min_v else img_arr / 255.0
        
        mask_arr = (np.array(mask, dtype=np.float32) > 128.0).astype(np.float32)
        
        itensor = torch.from_numpy(norm_img).unsqueeze(0).float()
        mtensor = torch.from_numpy(mask_arr).unsqueeze(0).float()
        
        return itensor, mtensor

def compute_dice_iou(logits, targets, smooth=1.0, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)
    
    intersection = (preds_flat * targets_flat).sum().item()
    total_sum = preds_flat.sum().item() + targets_flat.sum().item()
    
    dice = (2.0 * intersection + smooth) / (total_sum + smooth)
    
    union = total_sum - intersection
    iou = (intersection + smooth) / (union + smooth)
    
    return dice, iou

def train_unet(epochs: int = 8, batch_size: int = 32, lr: float = 1e-3):
    os.makedirs("models", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Fast Training UNet on Tumor-Positive Slices on device: {device}")
    
    train_dataset = RealMRISegmentationDataset(os.path.join("dataset", "train"), augment=True, positive_only=True)
    val_dataset = RealMRISegmentationDataset(os.path.join("dataset", "val"), augment=False, positive_only=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = LightweightUNet2D(in_channels=1, out_channels=1).to(device)
    criterion = DiceBCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    best_val_dice = 0.0
    model_save_path = os.path.join("models", "brain_tumor_unet_2d.pth")
    
    print(f"Beginning Fast Real LGG UNet Training ({epochs} Epochs, {len(train_dataset)} positive train, {len(val_dataset)} positive val)...")
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            
        train_loss /= len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_dice_total = 0.0
        val_iou_total = 0.0
        
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                loss = criterion(logits, masks)
                val_loss += loss.item() * imgs.size(0)
                
                dice, iou = compute_dice_iou(logits, masks)
                val_dice_total += dice * imgs.size(0)
                val_iou_total += iou * imgs.size(0)
                
        val_loss /= len(val_dataset)
        val_dice = val_dice_total / len(val_dataset)
        val_iou = val_iou_total / len(val_dataset)
        
        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f}")
        
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save({
                'state_dict': model.state_dict(),
                'val_dice': val_dice,
                'val_iou': val_iou
            }, model_save_path)
            
    total_time = time.time() - start_time
    print(f"\nUNet Training completed in {total_time:.2f} seconds.")
    print(f"Best Validation Dice Score: {best_val_dice:.4f}")
    print(f"Saved trained PyTorch UNet checkpoint to: {model_save_path}")

if __name__ == "__main__":
    train_unet()
