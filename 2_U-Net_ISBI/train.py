import os
import torch
import numpy as np
from model import UNet

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from skimage import io
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

# Kaggle路径设置
DATA_DIR = "/kaggle/input/d/yujuweng/week02/ISBI2012"
OUTPUT_DIR = "/kaggle/working"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "sample_results"), exist_ok=True)

# 数据预处理
class ISBIDataset(Dataset):
    def __init__(self, image_path, label_path, transform=None, train=True):
        self.images = io.imread(image_path)  # 确保是 [N, H, W]
        self.labels = io.imread(label_path)  # 确保是 [N, H, W]
        assert len(self.images) == len(self.labels)
        
        self.transform = transform
        self.train = train
        
        # 分割训练集和验证集（25训练 + 5验证）
        if train:
            self.images = self.images[:25]
            self.labels = self.labels[:25]
        else:
            self.images = self.images[25:]
            self.labels = self.labels[25:]
        
        print(f"{'Train' if train else 'Val'} set size: {len(self.images)}")  # 调试输出

    def __len__(self):
        return len(self.images)  # 必须实现

    def __getitem__(self, idx):
        image = self.images[idx].astype(np.float32) / 255.0  # [H, W]
        label = (self.labels[idx] > 0).astype(np.float32)    # [H, W]
        
        if self.transform:
            image = self.transform(image)  # ToTensor() -> [1, H, W]
            label = self.transform(label)  # ToTensor() -> [1, H, W]
        
        return image, label


def save_sample_results(epoch, model, val_loader, device):
    model.eval()
    with torch.no_grad():
        sample_images, sample_labels = next(iter(val_loader))
        sample_images = sample_images.to(device)
        preds = torch.sigmoid(model(sample_images))
        preds = (preds > 0.5).float()
        
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        for i in range(3):
            axes[i, 0].imshow(sample_images[i].cpu().squeeze(), cmap='gray')
            axes[i, 0].set_title('Input Image')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(sample_labels[i].cpu().squeeze(), cmap='gray')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(preds[i].cpu().squeeze(), cmap='gray')
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')
        
        plt.suptitle(f'Epoch {epoch+1} Results', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'sample_results', f'epoch_{epoch+1}.png'), bbox_inches='tight')
        plt.close()

def train_model():
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 数据加载
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = ISBIDataset(
        os.path.join(DATA_DIR, "train-volume.tif"),
        os.path.join(DATA_DIR, "train-labels.tif"),
        transform,
        train=True
    )
    
    val_dataset = ISBIDataset(
        os.path.join(DATA_DIR, "train-volume.tif"),
        os.path.join(DATA_DIR, "train-labels.tif"),
        transform,
        train=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # 模型初始化
    model = UNet().to(device)
    criterion = nn.BCEWithLogitsLoss()  # 二分类任务使用BCEWithLogitsLoss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    
    # 训练循环
    num_epochs = 20
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0
        
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            images = images.to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        # 验证集评估
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                epoch_val_loss += loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        # 保存样本结果
        if (epoch+1) % 5 == 0 or epoch == 0:
            save_sample_results(epoch, model, val_loader, device)
        
        # 学习率调整
        scheduler.step(avg_val_loss)
    
    # 保存损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(OUTPUT_DIR, 'loss_curve.png'))
    plt.close()
    
    # 保存模型
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'trained_model.pth'))
    print(f'Training complete. Model and results saved to {OUTPUT_DIR}')

if __name__ == '__main__':
    train_model()
