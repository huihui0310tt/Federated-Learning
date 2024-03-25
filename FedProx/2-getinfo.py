from os import listdir, path
import os
import json
from tabulate import tabulate

def count_files_in_directory(directory):
    """
    計算目錄中的檔案數量
    """
    if not path.exists(directory) or not path.isdir(directory):
        print(f"目錄 {directory} 不存在或不是一個有效的目錄。")
        return 0

    file_count = 0

    # 遞迴地計算目錄中的檔案數量
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_count += 1

    return file_count

def main():
    """
    主程式邏輯
    """
    # 設定目錄路徑
    base_directory = "./data"

    # 讀取 configure.json 檔案
    with open("configure.json", "r") as f:
        config = json.load(f)

    # 從配置中獲取使用者名稱和子目錄
    users = config["User"]
    users.append('test')
    subdirectories = config["DataCategory"]

    # 儲存結果的字典，以子目錄為鍵，使用者為值
    results = {}

    # 遍歷每個使用者目錄並儲存結果
    for user in users:
        for sub_directory in subdirectories:
            sub_directory_path = path.join(base_directory, user, sub_directory)
            if path.isdir(sub_directory_path):
                sub_directory_file_count = count_files_in_directory(sub_directory_path)
                # 檢查子目錄是否已經存在於字典中，如果不存在，則創建一個新的空列表
                if sub_directory not in results:
                    results[sub_directory] = [None] * len(users)
                # 在子目錄對應的列表中的使用者位置儲存檔案數量
                user_index = users.index(user)
                results[sub_directory][user_index] = sub_directory_file_count

    # 將字典轉換成列表形式，以便 tabulate 函式使用
    table_data = [[sub_directory] + file_counts for sub_directory, file_counts in results.items()]

    # 輸出結果
    print(tabulate(table_data, headers=["Subdirectory"] + users))

if __name__ == "__main__":
    main()
