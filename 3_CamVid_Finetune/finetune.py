import os 
import torch as torch
import torchviz
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision.models import resnet50
import torchvision.models.segmentation as segmentation 
from torchviz import make_dot
from torch.autograd import Variable
from sklearn.metrics import jaccard_score
from torch.optim.lr_scheduler import ReduceLROnPlateau  
from torch.optim import lr_scheduler

import pandas as pd

def load_color_mapping(csv_path):
    # Load the CSV file
    df = pd.read_csv(csv_path)
    # Create a mapping from (r, g, b) to the class index
    color_mapping = {(row['r'], row['g'], row['b']): idx for idx, row in df.iterrows()}
    return color_mapping,df


color_mapping,df=load_color_mapping("/kaggle/input/camvid/CamVid/class_dict.csv")
print(color_mapping)

# 数据集
class CamVidDataset(Dataset):
    def __init__(self, image_dir, label_dir, color_mapping, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.color_mapping = color_mapping
        self.transform = transform
        self.image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg')])
        self.label_files = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load the image and label
        image = cv2.imread(self.image_files[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        label = cv2.imread(self.label_files[idx]) # Load as RGB
        label=cv2.cvtColor(label,cv2.COLOR_BGR2RGB)
        #Convert the RGB label to class indices
        label = self.convert_rgb_to_class(label)
#         Apply transformations, if any
        if self.transform:
            augmented = self.transform(image=image, mask=label)
            image = augmented['image']
            label = augmented['mask']

        return image, label.long()

    def convert_rgb_to_class(self, label):
        """Convert RGB mask to class indices."""
        # Create an empty mask with the same shape as the label
        class_mask = np.zeros((label.shape[0], label.shape[1]), dtype=int)

        # Iterate over each pixel and assign the class index based on RGB value
        for rgb, class_idx in self.color_mapping.items():
            mask = (label[:, :, 0] == rgb[0]) & (label[:, :, 1] == rgb[1]) & (label[:, :, 2] == rgb[2])
            class_mask[mask] = class_idx

        return class_mask

# 数据增强
def get_transforms(train=True):
    if train:
        return A.Compose([
            A.Resize(400, 520),  
            A.RandomCrop(height=352, width=480),  
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GridDistortion(p=0.2),  # 增强形变鲁棒性
            A.Rotate(limit=15, p=0.5), 
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            A.Normalize(mean=(0.390, 0.405, 0.414), std=(0.274, 0.285, 0.297)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(352, 480), 
            A.Normalize(mean=(0.390, 0.405, 0.414), std=(0.274, 0.285, 0.297)),
            ToTensorV2()
        ])


# Instantiate datasets with the color map
train_dataset = CamVidDataset('/kaggle/input/camvid/CamVid/train', '/kaggle/input/camvid/CamVid/train_labels', color_mapping=color_mapping, transform=get_transforms(train=True))
val_dataset = CamVidDataset('/kaggle/input/camvid/CamVid/val', '/kaggle/input/camvid/CamVid/val_labels', color_mapping=color_mapping, transform=get_transforms(train=False))
test_dataset = CamVidDataset('/kaggle/input/camvid/CamVid/test', '/kaggle/input/camvid/CamVid/test_labels', color_mapping=color_mapping, transform=get_transforms(train=False))

    
# 加载器
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)

# deeplab3模型

class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes, freeze_backbone=True):
        super(DeepLabV3Plus, self).__init__()
        # Load the pre-trained DeepLabV3+ model
        self.model = segmentation.deeplabv3_resnet101(pretrained=True)
        
        # Modify the classifier to output num_classes
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1))
        self.model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1))

                
        # 确保所有参数都在同一设备上
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
        # Freeze backbone if needed
        if freeze_backbone:
            self._freeze_backbone()
            
    
    def _freeze_backbone(self):
        """Freeze all backbone parameters except classification heads"""
        # Freeze the entire backbone
        for param in self.model.backbone.parameters():
            param.requires_grad = False
        
        # Freeze ASPP module parameters
        for param in self.model.classifier[0].parameters():
            param.requires_grad = False
        
        # Ensure classification heads are trainable
        for param in self.model.classifier[4].parameters():  # Final conv layer
            param.requires_grad = True
        for param in self.model.aux_classifier[4].parameters():  # Auxiliary head
            param.requires_grad = True
    
    def _unfreeze_all(self):
        """Unfreeze all parameters"""
        for param in self.model.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        x = x.to(self.device)
        return self.model(x)['out']

# 训练


class EarlyStopping:
    def __init__(self, patience=7, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        
    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.val_loss_min = val_loss
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.val_loss_min = val_loss
            self.counter = 0

import torch.nn as nn
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        inputs = torch.softmax(inputs, dim=1)
        targets = nn.functional.one_hot(targets.long(), inputs.shape[1]).permute(0,3,1,2).float()
        intersection = (inputs * targets).sum(dim=(2,3))
        union = inputs.sum(dim=(2,3)) + targets.sum(dim=(2,3))
        dice = (2.*intersection + self.smooth)/(union + self.smooth)
        return 1 - dice.mean()

class JaccardLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        inputs = torch.softmax(inputs, dim=1)
        targets = nn.functional.one_hot(targets.long(), inputs.shape[1]).permute(0,3,1,2).float()
        intersection = (inputs * targets).sum(dim=(2,3))
        union = inputs.sum(dim=(2,3)) + targets.sum(dim=(2,3)) - intersection
        jaccard = (intersection + self.smooth)/(union + self.smooth)
        return 1 - jaccard.mean()

class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.6, jaccard_weight=0.2):
        super().__init__()
        self.dice_weight = dice_weight
        self.jaccard_weight = jaccard_weight
        self.ce_weight = 1 - dice_weight - jaccard_weight
        assert abs(self.dice_weight + self.jaccard_weight + self.ce_weight - 1.0) < 1e-6
        self.dice = DiceLoss()
        self.jaccard = JaccardLoss()
        self.ce = nn.CrossEntropyLoss()
    
    def forward(self, inputs, targets):
        return (self.dice_weight * self.dice(inputs, targets) +
                self.jaccard_weight * self.jaccard(inputs, targets) +
                self.ce_weight * self.ce(inputs, targets.long()))


class TwoPhaseTrainer:
    def __init__(self, num_classes, device, train_loader, val_loader):
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        
        print("Initializing model...")
        self.model = DeepLabV3Plus(num_classes=num_classes, freeze_backbone=True)
        self._convert_bn_to_gn(self.model).to(device)
        
        self.criterion = CombinedLoss(dice_weight=0.6, jaccard_weight=0.2).to(device)
        
        # Phase 1: Head only
        phase1_params = list(self.model.model.classifier[4].parameters()) + \
                       list(self.model.model.aux_classifier[4].parameters())
        self.phase1_optimizer = Adam(phase1_params, lr=5e-4, weight_decay=1e-5)
        self.phase1_scheduler = ReduceLROnPlateau(self.phase1_optimizer, mode='min', 
                                                patience=2, factor=0.5, verbose=True)
        self.phase1_early_stopping = EarlyStopping(patience=5, delta=0.001)
        
        # Phase 2: Full model
        self.phase2_optimizer = Adam(self.model.parameters(), lr=3e-4, weight_decay=1e-4)
        self.phase2_scheduler = ReduceLROnPlateau(self.phase2_optimizer, mode='min',
                                                patience=2, factor=0.5, verbose=True)
        self.phase2_early_stopping = EarlyStopping(patience=5, delta=0.001)
        
        self.history = {'phase1': {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]},
                       'phase2': {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}}

    def _convert_bn_to_gn(self, module):
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                num_groups = min(32, child.num_features)
                setattr(module, name, nn.GroupNorm(num_groups, child.num_features))
            else:
                self._convert_bn_to_gn(child)
        return module

    def _train_epoch(self, optimizer, phase='head'):
        self.model.train()
        epoch_loss, correct, total = 0, 0, 0
        
        for images, masks in tqdm(self.train_loader, desc=f'Training ({phase})'):
            images, masks = images.to(self.device), masks.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            
            with torch.no_grad():
                pred = torch.argmax(outputs, dim=1)
                correct += (pred == masks).sum().item()
                total += masks.numel()
                epoch_loss += loss.item()
        
        return epoch_loss/len(self.train_loader), correct/total

    def _validate(self):
        self.model.eval()
        val_loss, correct, total = 0, 0, 0
        
        with torch.no_grad():
            for images, masks in tqdm(self.val_loader, desc='Validating'):
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                pred = torch.argmax(outputs, dim=1)
                correct += (pred == masks).sum().item()
                total += masks.numel()
                val_loss += loss.item()
        
        return val_loss/len(self.val_loader), correct/total

    def _visualize_predictions(self, epoch, phase, num_samples=8):
        self.model.eval()
        indices = torch.randperm(len(self.val_loader.dataset))[:num_samples]
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
        
        for i, idx in enumerate(indices):
            img, mask = self.val_loader.dataset[idx]
            with torch.no_grad():
                pred = torch.argmax(self.model(img.unsqueeze(0).to(self.device)), 1).squeeze().cpu()
            
            axes[i,0].imshow(img.permute(1,2,0).cpu().numpy())
            axes[i,1].imshow(mask.numpy(), vmin=0, vmax=self.num_classes-1)
            axes[i,2].imshow(pred.numpy(), vmin=0, vmax=self.num_classes-1)
            [ax.axis('off') for ax in axes[i]]
        
        plt.savefig(f'predictions_epoch{epoch}_{phase}.png')
        plt.close()

    def train_phase1(self, num_epochs=20):
        print("\n=== Phase 1: Training Head ===")
        best_loss = float('inf')
        
        for epoch in range(1, num_epochs+1):
            train_loss, train_acc = self._train_epoch(self.phase1_optimizer)
            val_loss, val_acc = self._validate()
            
            self.history['phase1']['train_loss'].append(train_loss)
            self.history['phase1']['val_loss'].append(val_loss)
            self.history['phase1']['train_acc'].append(train_acc)
            self.history['phase1']['val_acc'].append(val_acc)
            
            self.phase1_scheduler.step(val_loss)
            self.phase1_early_stopping(val_loss)
            
            print(f'Epoch {epoch}/{num_epochs} | '
                  f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | '
                  f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
            
            self._visualize_predictions(epoch, 'phase1')
            
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.model, 'best_phase1_full.pth')
                print(f"Saved best model (loss: {val_loss:.4f})")
            
            if self.phase1_early_stopping.early_stop:
                print("Early stopping triggered")
                break

    def train_phase2(self, num_epochs=10):
        print("\n=== Phase 2: Fine-tuning ===")
        self.model._unfreeze_all()
        best_loss = float('inf')
        
        for epoch in range(1, num_epochs+1):
            train_loss, train_acc = self._train_epoch(self.phase2_optimizer, 'full')
            val_loss, val_acc = self._validate()
            
            self.history['phase2']['train_loss'].append(train_loss)
            self.history['phase2']['val_loss'].append(val_loss)
            self.history['phase2']['train_acc'].append(train_acc)
            self.history['phase2']['val_acc'].append(val_acc)
            
            self.phase2_scheduler.step(val_loss)
            self.phase2_early_stopping(val_loss)
            
            print(f'Epoch {epoch}/{num_epochs} | '
                  f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | '
                  f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
            
            self._visualize_predictions(epoch, 'phase2')
            
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.model, 'best_phase2_full.pth')
                print(f"Saved best model (loss: {val_loss:.4f})")
            
            if self.phase2_early_stopping.early_stop:
                print("Early stopping triggered")
                break

    def train(self):
        self.train_phase1()
        self.train_phase2()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trainer = TwoPhaseTrainer(num_classes=32, device=device, train_loader=train_loader, val_loader=val_loader)
trainer.train()

