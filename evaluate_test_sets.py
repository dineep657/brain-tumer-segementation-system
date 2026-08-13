import os
import torch
import numpy as np
from PIL import Image

from train_2d_unet import LightweightUNet2D, compute_dice_iou
from train_2d_classifier import LightweightClassifier2D, CLASSES

def evaluate_segmentation_test_set():
    print("=" * 70)
    print("  PHASE 4A: UNET SEGMENTATION EVALUATION ON REAL UNSEEN LGG TEST SET")
    print("=" * 70)
    
    checkpoint_path = os.path.join("models", "brain_tumor_unet_2d.pth")
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint missing at '{checkpoint_path}'")
        return 0.0, 0.0
        
    model = LightweightUNet2D(in_channels=1, out_channels=1)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
    model.eval()
    
    test_img_dir = os.path.join("dataset", "test", "images")
    test_mask_dir = os.path.join("dataset", "test", "masks")
    
    filenames = sorted([f for f in os.listdir(test_img_dir) if f.endswith('.png')])
    print(f"Loaded {len(filenames)} real unseen LGG test image-mask pairs.")
    
    total_dice = 0.0
    total_iou = 0.0
    valid_pairs = 0
    
    with torch.no_grad():
        for fname in filenames:
            ipath = os.path.join(test_img_dir, fname)
            mpath = os.path.join(test_mask_dir, fname)
            
            img = Image.open(ipath).convert('L').resize((128, 128), Image.Resampling.BILINEAR)
            mask = Image.open(mpath).convert('L').resize((128, 128), Image.Resampling.NEAREST)
            
            mask_arr = (np.array(mask, dtype=np.float32) > 128.0).astype(np.float32)
            if mask_arr.sum() == 0:
                continue # Only evaluate segmentation metrics on tumor-positive slices
                
            img_arr = np.array(img, dtype=np.float32)
            min_v, max_v = img_arr.min(), img_arr.max()
            norm_img = (img_arr - min_v) / (max_v - min_v + 1e-8) if max_v > min_v else img_arr / 255.0
            
            itensor = torch.from_numpy(norm_img).unsqueeze(0).unsqueeze(0).float()
            mtensor = torch.from_numpy(mask_arr).unsqueeze(0).unsqueeze(0).float()
            
            logits = model(itensor)
            dice, iou = compute_dice_iou(logits, mtensor)
            
            total_dice += dice
            total_iou += iou
            valid_pairs += 1
            
    avg_dice = total_dice / valid_pairs if valid_pairs > 0 else 0.0
    avg_iou = total_iou / valid_pairs if valid_pairs > 0 else 0.0
    
    print(f"  Evaluated Tumor Slices: {valid_pairs} positive pairs")
    print(f"  Real Test Dice Score  : {avg_dice:.4f} ({avg_dice*100:.2f}%)")
    print(f"  Real Test IoU Score   : {avg_iou:.4f} ({avg_iou*100:.2f}%)")
    return avg_dice, avg_iou

def evaluate_classification_test_set():
    print("\n" + "=" * 70)
    print("  PHASE 4B: CNN CLASSIFIER EVALUATION ON REAL UNSEEN KAGGLE TEST SET")
    print("=" * 70)
    
    checkpoint_path = os.path.join("models", "brain_tumor_classifier_2d.pth")
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint missing at '{checkpoint_path}'")
        return
        
    model = LightweightClassifier2D(num_classes=len(CLASSES))
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
    model.eval()
    
    test_base_dir = os.path.join("classification_dataset", "test")
    
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for c_idx, cname in enumerate(CLASSES):
            cdir = os.path.join(test_base_dir, cname)
            if not os.path.exists(cdir):
                continue
            for fname in sorted(os.listdir(cdir)):
                if fname.endswith('.png'):
                    fpath = os.path.join(cdir, fname)
                    img = Image.open(fpath).convert('L').resize((128, 128), Image.Resampling.BILINEAR)
                    img_arr = np.array(img, dtype=np.float32)
                    min_v, max_v = img_arr.min(), img_arr.max()
                    norm_img = (img_arr - min_v) / (max_v - min_v + 1e-8) if max_v > min_v else img_arr / 255.0
                    
                    itensor = torch.from_numpy(norm_img).unsqueeze(0).unsqueeze(0).float()
                    logits = model(itensor)
                    pred_idx = logits.argmax(dim=1).item()
                    
                    y_true.append(c_idx)
                    y_pred.append(pred_idx)
                    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    num_classes = len(CLASSES)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
        
    acc = np.mean(y_true == y_pred)
    
    precisions = []
    recalls = []
    f1s = []
    
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        
        print(f"  Class [{CLASSES[i]:<10}]: Precision={prec:.4f} | Recall={rec:.4f} | F1={f1:.4f}")
        
    macro_f1 = np.mean(f1s)
    macro_prec = np.mean(precisions)
    macro_rec = np.mean(recalls)
    
    print("\n----------------------------------------------------------------------")
    print(f"  Real Test Images Evaluated : {len(y_true)} (400 per class)")
    print(f"  Overall Test Accuracy      : {acc*100:.2f} %")
    print(f"  Macro Precision            : {macro_prec:.4f}")
    print(f"  Macro Recall               : {macro_rec:.4f}")
    print(f"  Macro F1-Score             : {macro_f1:.4f}")
    print("\n--- 4x4 Confusion Matrix ---")
    print("Columns = Predicted | Rows = Ground Truth")
    print(f"Classes: {CLASSES}")
    print(cm)
    return acc, macro_prec, macro_rec, macro_f1, cm

if __name__ == "__main__":
    evaluate_segmentation_test_set()
    evaluate_classification_test_set()
