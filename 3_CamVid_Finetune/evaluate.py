import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from finetune import CamVidDataset, COLOR_MAP, image_transform, label_to_index, NUM_CLASSES

model = torch.load(
    '/kaggle/input/bestphase2/best_phase2_full.pth', 
    map_location='cpu',
    weights_only=False  # 允许加载完整模型
)
model.eval()

class CamVidDataset(Dataset):
    def __init__(self, image_dir, label_dir, color_mapping, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.color_mapping = color_mapping
        self.transform = transform
        
        # 获取所有图片文件
        self.image_files = sorted([
            f for f in os.listdir(image_dir) 
            if f.endswith(('.png', '.jpg'))
        ])
        
        # 构建标签文件名映射
        label_files = os.listdir(label_dir)
        self.label_map = {}
        for lbl in label_files:
            if lbl.endswith('.png'):
                base_name = lbl.replace('_L.png', '.png')
                self.label_map[base_name] = lbl
        
        # 验证匹配
        self.valid_image_files = []
        for img in self.image_files:
            if img in self.label_map:
                self.valid_image_files.append(img)
            else:
                print(f"警告: 图像 {img} 没有对应的标签文件")
        
        assert len(self.valid_image_files) > 0, "没有找到有效的图像-标签对！"

    def __len__(self):
        return len(self.valid_image_files)

    def __getitem__(self, idx):
        img_name = self.valid_image_files[idx]
        lbl_name = self.label_map[img_name]
        
        # 加载图像和标签
        image = cv2.imread(os.path.join(self.image_dir, img_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(os.path.join(self.label_dir, lbl_name))
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        
        # 转换标签为类别索引
        label = self.convert_rgb_to_class(label)
        
        # 数据增强
        if self.transform:
            augmented = self.transform(image=image, mask=label)
            image = augmented['image']
            label = augmented['mask']
            
            # 统一处理标签类型
            if isinstance(label, torch.Tensor):
                label = label.long()
            else:
                label = torch.from_numpy(label).long()
        else:
            # 如果没有transform，手动转换
            image = torch.from_numpy(image).float().permute(2, 0, 1)
            label = torch.from_numpy(label).long()
        
        return image, label

    def convert_rgb_to_class(self, label):
        class_mask = np.zeros((label.shape[0], label.shape[1]), dtype=np.int64)
        for rgb, class_idx in self.color_mapping.items():
            mask = np.all(label == np.array(rgb), axis=-1)
            class_mask[mask] = class_idx
        return class_mask

    # 1. 首先确定设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. 加载模型时指定map_location
model = torch.load(
    '/kaggle/input/bestphase2/best_phase2_full.pth', 
    map_location=device,  # 直接加载到目标设备
    weights_only=False
)

# 3. 将模型设置为评估模式并转移到设备
model = model.to(device)
model.eval()

# ==================== 定义评估指标 ====================
def pixel_accuracy(pred, label):
    """计算像素准确率"""
    correct = (pred == label).sum().item()
    total = label.numel()
    return correct / total

def mIoU(pred, label, n_classes):
    """计算mIoU和各类别IoU"""
    ious = []
    for cls in range(n_classes):
        pred_cls = pred == cls
        label_cls = label == cls
        intersection = (pred_cls & label_cls).sum().float()
        union = (pred_cls | label_cls).sum().float()
        iou = (intersection + 1e-6) / (union + 1e-6)  # 平滑处理避免除零
        ious.append(iou.item())
    return torch.mean(torch.tensor(ious)), ious

# ==================== 数据加载 ====================
color_mapping = {
    (64, 128, 64): 0, (192, 0, 128): 1, (0, 128, 192): 2, (0, 128, 64): 3, 
    (128, 0, 0): 4, (64, 0, 128): 5, (64, 0, 192): 6, (192, 128, 64): 7, 
    (192, 192, 128): 8, (64, 64, 128): 9, (128, 0, 192): 10, (192, 0, 64): 11, 
    (128, 128, 64): 12, (192, 0, 192): 13, (128, 64, 64): 14, (64, 192, 128): 15, 
    (64, 64, 0): 16, (128, 64, 128): 17, (128, 128, 192): 18, (0, 0, 192): 19, 
    (192, 128, 128): 20, (128, 128, 128): 21, (64, 128, 192): 22, (0, 0, 64): 23, 
    (0, 64, 64): 24, (192, 64, 128): 25, (128, 128, 0): 26, (192, 128, 192): 27, 
    (64, 0, 64): 28, (192, 192, 0): 29, (0, 0, 0): 30, (64, 192, 0): 31}

test_transform = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

test_dataset = CamVidDataset(
    image_dir="/kaggle/input/camvid/CamVid/test",
    label_dir="/kaggle/input/camvid/CamVid/test_labels",
    color_mapping=color_mapping,
    transform=test_transform
)

test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# ==================== 模型评估 ====================
total_pixel_acc = 0.0
total_miou = 0.0
class_iou = {i: 0.0 for i in range(32)}

with torch.no_grad():
    # 添加tqdm进度条
    progress_bar = tqdm(test_loader, desc="Evaluating", unit="batch")
    
    for images, labels in progress_bar:
        # 将数据转移到与模型相同的设备
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        preds = outputs.argmax(1)
        
        # 计算指标
        batch_pixel_acc = pixel_accuracy(preds, labels)
        batch_miou, iou_list = mIoU(preds, labels, n_classes=32)
        
        total_pixel_acc += batch_pixel_acc
        total_miou += batch_miou
        for i in range(32):
            class_iou[i] += iou_list[i]
        
        # 更新进度条描述
        progress_bar.set_postfix({
            'PixelAcc': f"{batch_pixel_acc:.3f}",
            'mIoU': f"{batch_miou:.3f}"
        })

# 计算平均指标
avg_pixel_acc = total_pixel_acc / len(test_loader)
avg_miou = total_miou / len(test_loader)
for i in range(32):
    class_iou[i] /= len(test_loader)

# 打印结果
print("\nFinal Evaluation Results:")
print(f"Pixel Accuracy: {avg_pixel_acc:.4f}")
print(f"mIoU: {avg_miou:.4f}")
print("\nPer-class IoU:")
for cls_idx, iou in class_iou.items():
    print(f"  Class {cls_idx}: {iou:.4f}")

    
