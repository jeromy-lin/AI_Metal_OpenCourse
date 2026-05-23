# ===========================================================
# 主題：MetaDefect-CNN 金屬表面瑕疵視覺辨識系統 (使用CNN)
# 目標：使學員了解深度學習進行多類別工業瑕疵檢測的方法與概念
#     : 對比YOLO 請 學員思考與CNN的使用差異
# 作者：國立雲林科技大學電機系 林家仁
# ===========================================================

import os
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# 1. 自動解壓縮與路徑防呆
DATASET_PATH = "/content/NEU_Dataset"
zip_files = [f for f in os.listdir('/content') if f.endswith('.zip') and not f.startswith('.')]

if len(zip_files) == 0:
    raise FileNotFoundError("錯誤：找不到 .zip 檔案！請確認已將資料集上傳至 Colab 左側目錄。")

target_zip = os.path.join('/content', zip_files[0])
print(f"偵測到壓縮檔：{target_zip}，正在解壓縮...")
!unzip -o -q "{target_zip}" -d {DATASET_PATH}

if not os.path.exists(DATASET_PATH) or len(os.listdir(DATASET_PATH)) == 0:
    raise ValueError("解壓縮失敗，請確認 ZIP 檔案是否完整。")
print("解壓縮完成。")

# 處理雙層資料夾結構
subdirs = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d)) and not d.startswith('.')]
if len(subdirs) == 1 and any(os.path.isdir(os.path.join(DATASET_PATH, subdirs[0], s)) for s in os.listdir(os.path.join(DATASET_PATH, subdirs[0]))):
    DATASET_PATH = os.path.join(DATASET_PATH, subdirs[0])
    print(f"資料載入路徑已修正為: {DATASET_PATH}")

# 【優化點】微調影像大小至 96x96，大幅降低運算複雜度
IMG_SIZE = 96  
BATCH    = 32
EPOCHS   = 15   

# 2. 載入並切分資料集 (70% Train, 15% Val, 15% Test)
print("\n正在載入影像...")
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
print("瑕疵類別：", class_names)

val_test_batches = tf.data.experimental.cardinality(ds_val_test).numpy()
val_batches = val_test_batches // 2

ds_val  = ds_val_test.take(val_batches)
ds_test = ds_val_test.skip(val_batches)

# 3. 高效能資料流管線優化
AUTOTUNE = tf.data.AUTOTUNE
# 【優化點】移除了 map(preprocess)，改用 cache 阻斷磁碟重複讀取瓶頸
train_ds = ds_train.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds   = ds_val.cache().prefetch(AUTOTUNE)
test_ds  = ds_test.cache().prefetch(AUTOTUNE)

# 4. 顯示金屬瑕疵影像範例
plt.figure(figsize=(12, 5))
for images, labels in train_ds.take(1):
    for i in range(8):
        ax = plt.subplot(2, 4, i + 1)
        plt.imshow(images[i].numpy().astype("uint8")) # 轉回 uint8 配合顯示
        lbl = class_names[labels[i]]
        plt.title(lbl, fontsize=10, color="darkred")
        plt.axis("off")
plt.suptitle("NEU-CLS Metal Defect Examples", fontsize=14)
plt.tight_layout()
plt.show()

# 5. 建立 6 分類 CNN 模型
model = models.Sequential([
    # 【優化點】直接在模型第一層使用 GPU 進行像素正規化 (0~255 -> 0~1)
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
print(f"\n測試集準確度 (Test Accuracy): {test_acc:.3f}")

plt.figure(figsize=(6, 4))
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Val")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Curve")
plt.legend()
plt.grid(True)
plt.show()

# 8. 預測示範與成果視覺化 (使用完全獨立的測試集 test_ds)
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
