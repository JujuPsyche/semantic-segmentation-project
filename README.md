# semantic-segmentation-project

# Semantic Segmentation Project

## Overview

This project explores semantic segmentation through a progressive pipeline, covering multiple models and application domains:

- DeepLabV3 on PASCAL VOC 2012 (pretrained model inference)
- U-Net implemented from scratch for biomedical image segmentation (ISBI dataset)
- Fine-tuning DeepLabV3+ on the CamVid urban scene dataset

The project demonstrates a complete workflow from data preprocessing and model design to training optimization and performance evaluation.

---

## Key Features

- Implementation of U-Net from scratch (encoder-decoder with skip connections)
- Two-phase fine-tuning strategy for DeepLabV3+
- Advanced loss design (Dice + Jaccard + CrossEntropy)
- Data augmentation using Albumentations
- Detailed evaluation including Pixel Accuracy, mIoU, and class-wise IoU
- Error analysis with visualization (FN/FP heatmaps)

---

## Results

### CamVid (DeepLabV3+ Fine-tuning)

- Pixel Accuracy: **0.892**
- mIoU: **0.763**

The model performs well on large structured objects (e.g., road, sky, building), while smaller or less frequent classes remain challenging.

---

## Project Structure

- `deeplabv3-voc` – Pretrained model inference on PASCAL VOC
- `unet-medical` – U-Net implementation and training on ISBI dataset
- `deeplabv3plus-camvid` – Fine-tuning and evaluation on CamVid dataset

---

## Skills & Techniques

- PyTorch
- Deep Learning (CNN, Semantic Segmentation)
- Model training and fine-tuning
- Data preprocessing and augmentation
- Performance evaluation (IoU, Pixel Accuracy)

---

## Note

The detailed report is written in Chinese. Please refer to this README for an English summary.
