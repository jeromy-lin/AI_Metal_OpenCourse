# ==============================================================================
#  主題 : YOLOv8-clsSteelDefectModel.py
#  目標 : 以 NEU-CLS 進行金屬資料集之分類驗證 結合 YOLO V8
#  注意 : 已訓練完成模型並以其val資料夾中金屬瑕疵圖檔, 隨機取樣進行驗證
#  作者 : 國立雲林科技大學電機系 林家仁
# NEU-CLS Multi-Class Randomized Verification Pipeline (6 Classes)
# ==============================================================================

import os
import cv2
import glob
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# Configuration and Model Loading
# ------------------------------------------------------------------------------
print("====== [Inference] Searching for Trained Model Weights ======")

project_dir = 'NEU_Steel_Defects'
found_weights = glob.glob(f'./**/runs/classify/{project_dir}/**/weights/best.pt', recursive=True) + \
                glob.glob(f'./{project_dir}/**/weights/best.pt', recursive=True)

if not found_weights:
    found_weights = glob.glob('./**/weights/best.pt', recursive=True)

if found_weights:
    found_weights.sort(key=os.path.getmtime)
    best_weight_path = found_weights[-1]
    print(f"Success: Found trained weight at: {best_weight_path}")
else:
    raise FileNotFoundError("Error: 'best.pt' weight file could not be found.")

from ultralytics import YOLO
model = YOLO(best_weight_path)
print("Model loaded successfully.")

# ------------------------------------------------------------------------------
# Robust Class-Based Image Selection
# ------------------------------------------------------------------------------
print("\n====== [Inference] Sampling 1 Random Image Per Defect Class ======")

# Target 6 standard classes
target_classes = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

# Gather all potential validation image paths
all_val_images = []
for ext in ['/**/*.jpg', '/**/*.png', '/**/*.bmp', '/**/*.JPEG', '/**/*.PNG', '/**/*.BMP']:
    all_val_images.extend(glob.glob('./**/val' + ext, recursive=True))
    all_val_images.extend(glob.glob('./**/valid' + ext, recursive=True))

# Filter system and run logs
all_val_images = [img for img in all_val_images if 'runs' not in img and 'NEU_Steel_Defects' not in img]

# Classify found images into a dictionary based on their folder name
class_image_map = {cls: [] for cls in target_classes}
for img_path in all_val_images:
    folder_name = os.path.basename(os.path.dirname(img_path)).lower()
    for cls in target_classes:
        if cls in folder_name:
            class_image_map[cls].append(img_path)
            break

# Select exactly one random image per class
selected_samples = []
for cls in target_classes:
    pool = class_image_map[cls]
    if len(pool) == 0:
        print(f"Warning: No validation images found for class '{cls.upper()}'. Skipping.")
        continue
    random_sample = random.choice(pool)
    selected_samples.append((cls, random_sample))

if len(selected_samples) == 0:
    raise FileNotFoundError("Error: Failed to capture any class-specific images from the validation set.")

print(f"Successfully sampled {len(selected_samples)} target defect images.")

# ------------------------------------------------------------------------------
# Multi-Plot Inference and Visualization (2x3 Grid)
# ------------------------------------------------------------------------------
print("\n====== [Inference] Running Batch Diagnostics and Rendering Visualization ======")

# Create a 2 rows by 3 columns canvas
fig, axes = plt.subplots(2, 3, figsize=(18, 11), facecolor='white')
axes = axes.flatten()

for idx, (gt_class, img_path) in enumerate(selected_samples):
    ax = axes[idx]
    
    # Run prediction
    results = model.predict(source=img_path, imgsz=224, verbose=False)
    result = results[0]
    
    probs = result.probs
    top1_idx = probs.top1
    top1_conf = float(probs.top1conf)
    pred_class = result.names[top1_idx]
    
    # Load and process image
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Render image on subplot
    ax.imshow(img_rgb)
    ax.axis('off')
    
    # Set subtitle as the target filename or basic index
    ax.set_title(f"Sample {idx + 1}: {gt_class.upper()}", fontsize=12, fontweight='bold', pad=8)
    
    # Style text block dynamically based on accuracy (Green if correct, Red if wrong)
    is_correct = (gt_class.lower() == pred_class.lower())
    box_color = 'honeydew' if is_correct else 'mistyrose'
    text_color = 'darkgreen' if is_correct else 'darkred'
    border_style = dict(boxstyle='round,pad=0.4', facecolor=box_color, edgecolor=text_color, alpha=0.9)
    
    info_text = (
        f"GT: {gt_class.upper()}\n"
        f"PRED: {pred_class.upper()}\n"
        f"CONF: {top1_conf*100:.1f}%"
    )
    
    # Place text overlay on top of the image
    ax.text(8, 12, info_text, fontsize=10, fontweight='bold', color=text_color, bbox=border_style, va='top', ha='left')

# Hide unused axes if any class was missing
for empty_idx in range(len(selected_samples), len(axes)):
    axes[empty_idx].axis('off')

plt.suptitle('YOLOv8-cls Multi-Class Steel Defect Randomized Verification', fontsize=18, fontweight='bold', y=0.96)
plt.tight_layout(rect=[0, 0.03, 1, 0.93])
plt.show()

print("[Pipeline Execution Successful] Batch randomized validation completed.")
