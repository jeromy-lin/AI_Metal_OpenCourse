# ===========================================================
# 主題：MetaDefect-CNN 金屬表面瑕疵視覺辨識系統 (修正與優化版)
# 目標：使學員了解深度學習進行多類別工業瑕疵檢測的方法與效能評估
# 作者：國立雲林科技大學電機系 林家仁
# ===========================================================

import os
import zipfile
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns  # 引入繪圖庫

# 1. 自動解壓縮與資料夾層級精確定位
BASE_PATH = "/content/NEU_Dataset" 

zip_files = [f for f in os.listdir('/content') if f.endswith('.zip') and not f.startswith('.')]

if len(zip_files) == 0:
    raise FileNotFoundError("錯誤：在 /content 資料夾中找不到任何 .zip 檔案！請先將資料集上傳至 Colab。")

target_zip = os.path.join('/content', zip_files[0])
print(f"偵測到壓縮檔：{target_zip}，正在解壓縮...")
with zipfile.ZipFile(target_zip, 'r') as zip_ref:
    zip_ref.extractall(BASE_PATH)
print("解壓縮完成！")

# 自動尋找「真正包含多個瑕疵分類」的內層根目錄
DATASET_PATH = BASE_PATH
for root, dirs, files in os.walk(BASE_PATH):
    # 排除隱藏資料夾，當發現某個目錄下含有 3 個以上的子目錄（代表那是瑕疵類別層）
    valid_dirs = [d for d in dirs if not d.startswith('.')]
    if len(valid_dirs) >= 3:
        DATASET_PATH = root
        break

print(f" 系統自動鎖定的實際資料載入路徑為: {DATASET_PATH}")

# 【效能優化參數】調至 96x96 解析度 
IMG_SIZE = 96  
BATCH    = 32
EPOCHS   = 15   

# 2. 載入並切分資料集 (70% Train, 15% Val, 15% Test)
print("\n正在讀取影像並切分資料集...")
ds_train = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.3,
    subset="training",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    label_mode="int"  
)

ds_val_test = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.3,
    subset="validation",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    label_mode="int"
)

class_names = ds_train.class_names
num_classes = len(class_names)
print(" 成功辨識瑕疵類別：", class_names)

val_test_batches = tf.data.experimental.cardinality(ds_val_test).numpy()
val_batches = val_test_batches // 2

ds_val  = ds_val_test.take(val_batches)
ds_test = ds_val_test.skip(val_batches)

# 3. 資料流管線優化
AUTOTUNE = tf.data.AUTOTUNE
train_ds = ds_train.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds   = ds_val.cache().prefetch(AUTOTUNE)
test_ds  = ds_test.cache().prefetch(AUTOTUNE)

# 4. 顯示金屬瑕疵影像範例
plt.figure(figsize=(12, 5))
for images, labels in train_ds.take(1):
    for i in range(8):
        ax = plt.subplot(2, 4, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        lbl = class_names[labels[i]]
        plt.title(lbl, fontsize=10, color="darkred")
        plt.axis("off")
plt.suptitle("NEU-CLS Metal Defect Examples", fontsize=14)
plt.tight_layout()
plt.show()

# 5. 建立 6 分類 CNN 模型
model = models.Sequential([
    # 利用 GPU 進行即時影像歸一化 
    layers.Rescaling(1./255, input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),
    
    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),
    
    layers.Conv2D(128, 3, activation="relu"), 
    layers.MaxPooling2D(),
    
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),                       
    layers.Dense(num_classes, activation="softmax") 
])

model.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=["accuracy"])

model.summary()

# 6. 訓練模型
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds
)

# 7. 評估模型與學習曲線
test_loss, test_acc = model.evaluate(test_ds)
print(f"\n 測試集準確度 (Test Accuracy): {test_acc:.3f}")

plt.figure(figsize=(6, 4))
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Val")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Curve")
plt.legend()
plt.grid(True)
plt.show()

# 8. 預測示範與成果視覺化 (使用獨立測試集 test_ds)
for images, labels in test_ds.take(1):
    preds_prob = model.predict(images[:12])
    preds = tf.argmax(preds_prob, axis=1).numpy()
    
    plt.figure(figsize=(14, 5))
    for i in range(12):
        ax = plt.subplot(2, 6, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        
        true_lbl = class_names[labels[i]]
        pred_lbl = class_names[preds[i]]
        color = "green" if true_lbl == pred_lbl else "red"
        
        plt.title(f"T: {true_lbl}\nP: {pred_lbl}", color=color, fontsize=9)
        plt.axis("off")
    plt.suptitle("NEU-CLS Prediction vs Ground Truth (Test Dataset)", fontsize=14)
    plt.tight_layout()
    plt.show()

# 9. 產出正確的混淆矩陣 (Confusion Matrix)
print("\n正在計算測試集的混淆矩陣...")
y_true = []
y_pred = []

for images, labels in test_ds:
    preds_prob = model.predict(images, verbose=0)
    preds = tf.argmax(preds_prob, axis=1).numpy()
    y_true.extend(labels.numpy())
    y_pred.extend(preds)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix - NEU-CLS Test Dataset", fontsize=14, pad=15)
plt.ylabel("Actual Label (Ground Truth)", fontsize=12)
plt.xlabel("Predicted Label", fontsize=12)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
