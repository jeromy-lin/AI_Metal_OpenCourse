# =======================================================================
#  主題 : YOLOv8-clsSteelDefectModel.py
#  目標 : 以 NEU-CLS 進行金屬資料集之分類建模 結合 YOLO V8
#  注意 : 請預先下載 NEU-CLS.zip 資料集以利資料及訓練
#  作者 : 國立雲林科技大學電機系 林家仁
# NEU-CLS Steel Defect Classification Automated Training Workshop
# =====================================================================
import os
import cv2
import glob
import shutil
import zipfile
import torch
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# Step 1: Environment Installation and Verification
# ------------------------------------------------------------------------------
print("====== [Step 1/5] Installing and Checking Ultralytics YOLO Environment ======")
os.system('pip install ultralytics')

from ultralytics import YOLO, utils

device_id = 0 if torch.cuda.is_available() else 'cpu'
print(f"-> PyTorch Version: {torch.__version__}")
print(f"-> Target Compute Device: {device_id}")
utils.checks.check_yolo()


# ------------------------------------------------------------------------------
# Step 2: Unzip NEU-CLS.zip and Align YOLO Structure
# ------------------------------------------------------------------------------
print("\n====== [Step 2/5] Unzipping Dataset and Calibrating Structure ======")

zip_name = 'NEU-CLS.zip'
dst_root = './neu_cls_yolo'

if not os.path.exists(zip_name):
    raise FileNotFoundError(f"Error: {zip_name} not found in the current directory. Please upload the file first.")

# Clear old target directory to prevent data corruption
if os.path.exists(dst_root):
    shutil.rmtree(dst_root)

print(f"Extracting {zip_name} to {dst_root}...")
with zipfile.ZipFile(zip_name, 'r') as zip_ref:
    zip_ref.extractall(dst_root)
print("Unzip completed.")

# Rename 'valid' to 'val' to comply with YOLO classification requirements
old_valid_path = os.path.join(dst_root, 'valid')
new_val_path = os.path.join(dst_root, 'val')

if os.path.exists(old_valid_path):
    os.rename(old_valid_path, new_val_path)
    print("Structure Alignment: Successfully renamed 'valid' directory to 'val'.")
elif os.path.exists(new_val_path):
    print("Verification: 'val' directory exists. Structure is aligned.")
else:
    # Fallback mechanism if files are nested inside an extra folder
    sub_dirs = [d for d in os.listdir(dst_root) if os.path.isdir(os.path.join(dst_root, d))]
    if len(sub_dirs) == 1:
        inner_dir = os.path.join(dst_root, sub_dirs[0])
        print(f"Nested directory detected. Shifting search path to: {inner_dir}")
        if os.path.exists(os.path.join(inner_dir, 'valid')):
            os.rename(os.path.join(inner_dir, 'valid'), os.path.join(inner_dir, 'val'))
        dst_root = inner_dir

total_train_count = len(glob.glob(os.path.join(dst_root, 'train', '*', '*')))
print(f"YOLO Dataset Setup Complete. Total training images: {total_train_count}")


# ------------------------------------------------------------------------------
# Step 3: Load Pre-trained Weights and Execute Fast Training (Epoch=4)
# ------------------------------------------------------------------------------
print("\n====== [Step 3/5] Loading Weights and Launching YOLOv8-cls Training ======")
if total_train_count == 0:
    raise RuntimeError("Error: No images found in the 'train' folder. Please check the ZIP internal structure.")

model = YOLO('yolov8n-cls.pt')

# Launch Core Training Engine
results = model.train(
    data=dst_root,
    epochs=4,                         # 4 epochs for fast diagnostic run
    imgsz=224,                        
    batch=64,                         
    workers=0,                        
    device=device_id,                 
    project='NEU_Steel_Defects',      
    name='yolov8n_fast_run'           
)
print("YOLO Core Training Task Completed Successfully.")


# ------------------------------------------------------------------------------
# Step 4: Dynamic Path Localization and Diagnostic Visualization (English UI)
# ------------------------------------------------------------------------------
print("\n====== [Step 4/5] Locating Training Reports for Visual Diagnostics ======")

actual_save_dir = results.save_dir
print(f"Target Directory: {actual_save_dir}")

confusion_matrix_path = os.path.join(actual_save_dir, 'confusion_matrix.png')
results_chart_path = os.path.join(actual_save_dir, 'results.png')

# 1. Plot Confusion Matrix
if os.path.exists(confusion_matrix_path):
    img = cv2.imread(confusion_matrix_path)
    plt.figure(figsize=(9, 9), facecolor='white')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('YOLOv8-cls Steel Defect Confusion Matrix Diagnostics', fontsize=14, fontweight='bold', pad=15)
    plt.show()
else:
    print(f"Notice: Confusion matrix file not found at: {confusion_matrix_path}")

# 2. Plot Performance & Convergence Curves
if os.path.exists(results_chart_path):
    img = cv2.imread(results_chart_path)
    plt.figure(figsize=(11, 7), facecolor='white')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Training Metrics Monitoring (Loss & Accuracy Curves)', fontsize=14, fontweight='bold', pad=15)
    plt.show()
else:
    print(f"Notice: Convergence curves file not found at: {results_chart_path}")


# ------------------------------------------------------------------------------
# Step 5: Industrial Deployment Readiness (Export to ONNX)
# ------------------------------------------------------------------------------
print("\n====== [Step 5/5] Exporting Optimized Model to Industrial ONNX Format ======")
best_weight = os.path.join(actual_save_dir, 'weights', 'best.pt')

if os.path.exists(best_weight):
    optimized_model = YOLO(best_weight)
    onnx_output_path = optimized_model.export(format='onnx')
    print(f"\n[Pipeline Execution Successful]")
    print(f"-> PyTorch Best Weight (.pt): {best_weight}")
    print(f"-> Industrial Deployment Model (.onnx): {onnx_output_path}")
else:
    print(f"Error: Target weight file not found at: {best_weight}")
