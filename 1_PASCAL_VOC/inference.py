import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from pascal_voc_dataset import VOCSegmentationDataset
from torchvision.models.segmentation import deeplabv3_resnet101

def save_visualization(image, pred, gt, save_path):
    """保存可视化对比图：原图、真实标签、预测结果
    
    参数:
        image: 原始图像张量 [C, H, W]
        pred: 预测的分割掩码 [H, W]
        gt: 真实标签掩码 [H, W]
        save_path: 结果保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(18, 6))
    
    # 将标准化后的图像还原为可视格式
    image = image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    image = image.clamp(0, 1).permute(1, 2, 0).numpy()  # 格式：[C,H,W] ——> [H, W, 3]
    
    # 把数字标签转成彩色图
    pred_rgb = VOCSegmentationDataset.decode_segmap(pred)
    gt_rgb = VOCSegmentationDataset.decode_segmap(gt)
    
    # 绘制对比图
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(gt_rgb)
    plt.title("Ground Truth")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(pred_rgb)
    plt.title("Prediction")
    plt.axis('off')
    
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

def analyze_complex_scene(image, pred, gt, sample_id):
    """对复杂场景进行逐类错误分析并生成报告
    
    参数:
        image: 原始图像张量 [C, H, W]
        pred: 预测的分割掩码 [H, W]
        gt: 真实标签掩码 [H, W]
        sample_id: 样本ID（用于保存结果）
    """
    analysis_dir = f'results/complex/scene_analysis'
    os.makedirs(analysis_dir, exist_ok=True)
    
    # 归一化图像还原
    image = image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    image = image.clamp(0, 1).permute(1, 2, 0).numpy()
    
    # 逐类计算精准度
    report = []
    for class_id, name in enumerate(VOCSegmentationDataset.CLASS_NAMES):
        if class_id not in gt:
            continue
            
        # 计算 True Positive正确 / False Positive错误 / False Negative遗漏
        tp = np.sum((gt == class_id) & (pred == class_id))
        fp = np.sum((gt != class_id) & (pred == class_id))
        fn = np.sum((gt == class_id) & (pred != class_id))
        
        # 计算 Precision准确率 / Recall召回率 / IoU
        precision = tp / (tp + fp + 1e-10)  # 避免除零
        recall = tp / (tp + fn + 1e-10)
        iou = tp / (tp + fp + fn + 1e-10)
        
        report.append(f"{name}: Precision={precision:.2%}, Recall={recall:.2%}, IoU={iou:.2%}")
        
        # 生成错误热力图（红色 = FN，蓝色 = FP）
        plt.figure(figsize=(10, 5))
        plt.imshow(image)
        plt.imshow((gt == class_id) & (pred != class_id), alpha=0.5, cmap='Reds')
        plt.imshow((gt != class_id) & (pred == class_id), alpha=0.5, cmap='Blues')
        plt.title(f"{name} Errors (Red=FN, Blue=FP)")
        plt.axis('off')
        plt.savefig(f'{analysis_dir}/{name}_errors.png', bbox_inches='tight', dpi=150)
        plt.close()
    
    # 保存文本报告
    with open(f'{analysis_dir}/report.txt', 'w') as f:
        f.write("=== DeepLabV3 复杂场景分析报告 ===\n")
        f.write(f"样本ID: {sample_id}\n\n")
        f.write("各类别性能指标:\n\n")
        f.write("\n".join(report))

def main():
    """主函数：加载模型、运行推理并保存结果"""
    # 初始化设备（优先使用GPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建结果目录
    os.makedirs('results/simple', exist_ok=True)
    os.makedirs('results/complex', exist_ok=True)
    
    # 加载验证集（注意：crop_size需与训练时一致）
    dataset = VOCSegmentationDataset(
        root_dir="/Users/user1/Desktop/微软AI/week1",
        split='val',
        crop_size=512  # 图片统一裁剪成512x512
    )
    
    # 加载预训练的DeepLabV3模型
    model = deeplabv3_resnet101(weights='DEFAULT').to(device)
    model.eval()  # 切换到评估模式
    
    # 1. 处理5个简单样本
    simple_indices = [0, 5, 10, 15, 20]  
    for idx in simple_indices:
        image, gt = dataset[idx]  # image: [3, 512, 512], gt: [512, 512]
        
        with torch.no_grad():
            # 模型推理（输出尺寸为[1, 21, 64, 64]）
            output = model(image.unsqueeze(0).to(device))['out']
            
            # 获取预测类别（降维到[1, 64, 64]）
            pred = torch.argmax(output, dim=1)
            
            # 上采样到原始尺寸（最近邻插值保持类别索引）
            pred = torch.nn.functional.interpolate(
                pred.float().unsqueeze(1),  # 添加通道维度 [1, 1, 64, 64]
                size=image.shape[1:],       # 目标尺寸 [512, 512]
                mode='nearest'
            ).squeeze().cpu().numpy()       # 降维到 [512, 512]
        
        # 保存可视化结果
        save_visualization(
            image, pred, gt.numpy(),
            f'results/simple/sample_{idx}.png'
        )
    
    # 2. 处理1个复杂场景（选择包含多类别的样本）
    target_file = "2007_001311"  # 示例复杂场景文件名
    complex_idx = dataset.image_ids.index(target_file)
    image, gt = dataset[complex_idx]
    
    with torch.no_grad():
        # 同上处理流程
        output = model(image.unsqueeze(0).to(device))['out']
        pred = torch.argmax(output, dim=1)
        pred = torch.nn.functional.interpolate(
            pred.float().unsqueeze(1),
            size=image.shape[1:],
            mode='nearest'
        ).squeeze().cpu().numpy()
    
    # 保存复杂场景结果
    save_visualization(
        image, pred, gt.numpy(),
        'results/complex/scene.png'
    )
    
    # 执行详细分析
    analyze_complex_scene(
        image,
        pred,
        gt.numpy(),
        complex_idx
    )

if __name__ == "__main__":
    main()
