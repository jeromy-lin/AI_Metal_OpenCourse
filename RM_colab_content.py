# ===========================================================
#  主題 : RM_colab_content.py
#  目標 : 清除Colab Content 資料夾(避免暫存檔案影響計算) 
#  作者 : 國立雲林科技大學電機系 林家仁
# ============================================================

import os
import shutil

# 定義 Colab 的預設工作路徑
target_dir = '/content'

print("開始清理 Colab 工作目錄...")

for filename in os.listdir(target_dir):
    file_path = os.path.join(target_dir, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
            print(file_path, "已刪除")
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
            print(file_path, "資料夾及其內容已刪除")
    except Exception as e:
        print(f"無法刪除 {file_path}，原因: {e}")

print("--- 清理完成！目前目錄已完全清空 ---")
