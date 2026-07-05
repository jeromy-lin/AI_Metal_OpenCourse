# ===========================================================
# 主題：MetaDefect-CNN 金屬表面瑕疵視覺辨識系統 (工業級高精準度版)
# 升級：EfficientNetV2 + 192x192高解析度 + 動態學習率監控
# 作者：國立雲林科技大學電機系 林家仁
# ===========================================================

import os
import zipfile
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 1. 自動解壓縮與資料夾層級定位
BASE_PATH = "/content/NEU_Dataset" 

zip_files = [f for f in os.listdir('/content') if f.endswith('.zip') and not f.startswith('.')]

if len(zip_files) == 0:
    raise FileNotFoundError("錯誤：在 /content 資料夾中找不到任何 .zip 檔案。請先將資料集上傳至 Colab 環境。")

target_zip = os.path.join('/content', zip_files[0])
print(f"偵測到壓縮檔案：{target_zip}，正在執行解壓縮...")
with zipfile.ZipFile(target_zip, 'r') as zip_ref:
    zip_ref.extractall(BASE_PATH)
print("解壓縮完成。")

DATASET_PATH = os.path.join(BASE_PATH, 'NEU-CLS', 'train')

print(f"資料載入路徑鎖定為: {DATASET_PATH}")

# 超參數配置
IMG_SIZE = 192  # 提高解析度以完整保留微小紋理特徵
BATCH    = 32
EPOCHS   = 20   # 配合預訓練模型與動態學習率之收斂輪次

# 2. 載入影像並進行三維資料集切分 (70% Train, 15% Val, 15% Test)
print("\n正在讀取高解析度影像並配置資料集...")
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
print("成功識別之瑕疵類別：", class_names)

val_test_batches = tf.data.experimental.cardinality(ds_val_test).numpy()
val_batches = val_test_batches // 2

ds_val  = ds_val_test.take(val_batches)
ds_test = ds_val_test.skip(val_batches)

# 3. 高效能資料管線優化 (I/O 效能提升)
AUTOTUNE = tf.data.AUTOTUNE
train_ds = ds_train.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds   = ds_val.cache().prefetch(AUTOTUNE)
test_ds  = ds_test.cache().prefetch(AUTOTUNE)

# 4. 建構遷移學習網路架構 (EfficientNetV2-B0)
print("\n正在載入 EfficientNetV2-B0 預訓練模型底座...")

# 引入 EfficientNetV2 核心網路（排除頂層分類器，權重基於 ImageNet）
# 註：EfficientNet 架構內部自帶 Rescaling 機制，無需手動執行除以 255 之操作
base_model = tf.keras.applications.EfficientNetV2B0(
    include_top=False,
    weights='imagenet',
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    pooling='avg'
)

# 凍結特徵萃取層之權重，防止預訓練特徵破壞
base_model.trainable = False 

# 拼接自訂工業級全連接分類層
model = models.Sequential([
    base_model,
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),                       # 引入正則化以抑制過擬合
    layers.Dense(num_classes, activation="softmax") 
])

model.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=["accuracy"])

model.summary()

# 5. 配置動態學習率調整機制 (Callback)
# 若驗證集損失 (val_loss) 連續 3 個 Epoch 未調降，則將學習率衰減至目前的 0.5 倍
lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    verbose=1
)

# 6. 啟動模型訓練
print("\n執行模型訓練程序...")
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=[lr_callback]
)

# 7. 模型效能評估與學習曲線檢視
test_loss, test_acc = model.evaluate(test_ds)
print(f"\n獨立測試集最終準確度 (Test Accuracy): {test_acc:.3f}")

plt.figure(figsize=(6, 4))
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Val")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("EfficientNetV2 Training Curve")
plt.legend()
plt.grid(True)
plt.show()

# 8. 獨立測試集預測範例與成果視覺化
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
    plt.suptitle("EfficientNetV2 Prediction vs Ground Truth", fontsize=14)
    plt.tight_layout()
    plt.show()

# 9. 計算並產出定量評估混淆矩陣 (Confusion Matrix)
print("\n正在計算測試集混淆矩陣...")
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
plt.title("Confusion Matrix - EfficientNetV2 Industrial Level", fontsize=14, pad=15)
plt.ylabel("Actual Label (Ground Truth)", fontsize=12)
plt.xlabel("Predicted Label", fontsize=12)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
