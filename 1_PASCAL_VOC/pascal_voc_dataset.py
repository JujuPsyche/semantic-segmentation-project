import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import torchvision.transforms as transforms
import random

class VOCSegmentationDataset(Dataset):
    """
    PASCAL VOC 2012语义分割数据集加载器
    功能：
    1. 同步加载JPEGImages和SegmentationClass中的图像
    2. 应用相同的数据增强变换
    3. 图像归一化+标签编码
    """
    
    # VOC官方21类颜色映射（含背景）
    VOC_COLORMAP = [
        [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
        [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
        [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
        [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
        [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]
    ]

    CLASS_NAMES = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow', 'dining table', 'dog',
        'horse', 'motorbike', 'person', 'potted plant', 'sheep',
        'sofa', 'train', 'tv'
    ]

    def __init__(self, root_dir, split='train', crop_size=512):
        """
        参数:
            root_dir: 包含 VOCdevkit 的根目录
            split: 'train'或'val'
            crop_size: 裁剪尺寸
        """
        self.root_dir = os.path.join(root_dir, 'VOCdevkit/VOC2012')
        self.split = split
        self.crop_size = crop_size
        
        # 验证路径
        assert os.path.exists(self.root_dir), f"路径不存在: {self.root_dir}"
        
        # 加载划分文件
        with open(os.path.join(self.root_dir, 'ImageSets/Segmentation', f'{split}.txt')) as f:
            self.image_ids = [x.strip() for x in f.readlines()]
        
        # 构建路径列表
        self.image_paths = [os.path.join(self.root_dir, 'JPEGImages', f'{id}.jpg') for id in self.image_ids]
        self.label_paths = [os.path.join(self.root_dir, 'SegmentationClass', f'{id}.png') for id in self.image_ids]
        
        # 图像预处理
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # 读取图像和标签
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = Image.open(self.label_paths[idx]).convert('RGB')
        
        # 应用变换
        image, label = self._transform(image, label)
        
        # 转换为Tensor
        image = transforms.functional.to_tensor(image)
        image = self.normalize(image)
        label = self._encode_label(label)
        
        return image, label

    def _transform(self, img, mask):
        """ 对图像和标签应用相同的空间变换 """
        # 随机缩放
        scale = random.uniform(0.5, 2.0)
        new_w, new_h = int(img.width*scale), int(img.height*scale)
        
        # 调整大小
        img = transforms.functional.resize(img, (new_h, new_w), Image.BILINEAR)
        mask = transforms.functional.resize(mask, (new_h, new_w), Image.NEAREST)
        
        # 随机裁剪
        if min(new_h, new_w) < self.crop_size:
            padding = max(self.crop_size-new_w, self.crop_size-new_h)
            img = transforms.functional.pad(img, padding, fill=0)
            mask = transforms.functional.pad(mask, padding, fill=255)
        
        i, j, h, w = transforms.RandomCrop.get_params(img, (self.crop_size, self.crop_size))
        img = transforms.functional.crop(img, i, j, h, w)
        mask = transforms.functional.crop(mask, i, j, h, w)
        
        # 随机翻转
        if random.random() > 0.5:
            img = transforms.functional.hflip(img)
            mask = transforms.functional.hflip(mask)
            
        return img, mask

    def _encode_label(self, label_img):
        """ 将RGB标签转换为类别索引 """
        label = np.array(label_img)
        index_map = np.zeros(label.shape[:2], dtype=np.int64)
        
        # 处理边界
        index_map[np.all(label == [224, 224, 192], axis=2)] = 255
        
        # 颜色映射
        for idx, color in enumerate(self.VOC_COLORMAP):
            index_map[np.all(label == color, axis=2)] = idx
            
        return torch.from_numpy(index_map).long()

    @staticmethod
    def decode_segmap(index_map):
        """ 将预测结果转换回彩色图像 """
        rgb = np.zeros((*index_map.shape, 3))
        for idx, color in enumerate(VOCSegmentationDataset.VOC_COLORMAP):
            rgb[index_map == idx] = np.array(color) / 255
        rgb[index_map == 255] = [1, 1, 1]  # 边界显示为白色
        return rgb
