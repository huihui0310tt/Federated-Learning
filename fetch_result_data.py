import os
import csv
from datetime import datetime

# 定義關鍵字列表
keywords = ["ResNet", "CIFAR"]

def transpose_list(lst):
    return list(map(list, zip(*lst)))

def get_last_column_of_accuracy_csv_files():
    # 取得當前目錄路徑
    current_directory = os.getcwd()
    print("當前目錄:", current_directory)

    # 產生帶有時間戳記的檔案名稱
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_file_name = "accuracy_data_{}.csv".format(timestamp)
    output_file_path = os.path.join(current_directory, output_file_name)

    with open(output_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # 取得當前目錄中所有的目錄
        directories = [d for d in os.listdir(current_directory) if os.path.isdir(os.path.join(current_directory, d))]

        # 篩選出包含指定關鍵字的目錄並進行排序
        filtered_directories = sorted([d for d in directories if all(keyword in d for keyword in keywords)])

        if filtered_directories:
            print("包含指定關鍵字的目錄有:")
            for directory in filtered_directories:
                accuracy_data = [directory]
                result_save_path = os.path.join(current_directory, directory, "result_save")
                if os.path.exists(result_save_path):
                    # 列出/result_save資料夾中所有以"accuracy.csv"為結尾的檔案
                    accuracy_files = [f for f in os.listdir(result_save_path) if f.endswith("accuracy.csv")]
                    if accuracy_files:
                        for accuracy_file in accuracy_files:
                            accuracy_file_path = os.path.join(result_save_path, accuracy_file)
                            with open(accuracy_file_path, 'r') as file:
                                reader = csv.reader(file)
                                for row in reader:
                                    last_column = row[-1]
                                    accuracy_data.append(last_column)
                    else:
                        accuracy_data.append("none")
                else:
                    accuracy_data.append("none")
                # 寫入CSV檔案
                writer.writerow(accuracy_data)

    print("資料已寫入至 {} 檔案中。".format(output_file_path))

if __name__ == "__main__":
    get_last_column_of_accuracy_csv_files()
