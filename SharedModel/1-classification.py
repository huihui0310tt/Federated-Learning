import shutil
from os import listdir, makedirs
from os.path import isdir, join
import json

# 讀取配置檔案
file_path = 'configure.json'
with open(file_path, 'r') as file:
    json_data = json.load(file)

# 提取配置信息
dataset_name = json_data['DatasetName']
users = json_data['User']
labels = json_data['DataCategory']
data_distribution = json_data['DataDistribution']

# 建立用戶到分配數據的映射
user_distribution_map = {}
for dist in data_distribution:
    for key, value in dist.items():
        user_distribution_map[key] = value

# 找到 'shared' 的分配數據
shared_data = user_distribution_map.get("shared", None)
if shared_data is None:
    raise ValueError("配置中缺少 'shared' 數據")

# 設定原始資料和共享資料的路徑
if dataset_name == 'CIFAR':
    src = '../CIFAR_origindata'
elif dataset_name == 'COVID':
    src = '../COVID_origindata'
else:
    raise ValueError("未知的數據集名稱")

shared_path = "./data/shared"

# 確保共享資料夾和目標目錄存在
if not isdir(shared_path):
    makedirs(shared_path)

for label in labels:
    label_path = join(shared_path, label)
    if not isdir(label_path):
        makedirs(label_path)

# 將共享資料複製到共享資料夾
for label_idx, label_name in enumerate(labels):
    count = 0
    for file_name in listdir(join(src, label_name)):
        if count < shared_data[label_idx]:
            shutil.copy(join(src, label_name, file_name), join(shared_path, label_name, file_name))
            count += 1

# 確保每個用戶的資料夾存在
for user in users:
    user_path = join("./data", user)
    if not isdir(user_path):
        makedirs(user_path)
    for label_name in labels:
        label_path = join(user_path, label_name)
        if not isdir(label_path):
            makedirs(label_path)

# 分配剩餘資料給每個用戶
for label_idx, label_name in enumerate(labels):
    count = shared_data[label_idx]
    for file_name in listdir(join(src, label_name)):
        temp = shared_data[label_idx]
        # 確保用戶在 user_distribution_map 中存在
        for user_idx, user in enumerate(users):
            if user not in user_distribution_map:
                continue  # 跳過沒有分配數據的用戶
            user_dist = user_distribution_map[user]
            temp += user_dist[label_idx]
            if count < temp:
                shutil.copy(join(src, label_name, file_name), join("./data", user, label_name, file_name))
                count += 1
                break
