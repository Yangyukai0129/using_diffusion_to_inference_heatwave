# data_utils.py
import torch
from torch.utils.data import Dataset

class WeatherDataset(Dataset):
    def __init__(self, cond_data, target_data):
        self.cond_data = cond_data
        self.target_data = target_data

    def __len__(self):
        return len(self.cond_data)

    def __getitem__(self, idx):
        return self.cond_data[idx], self.target_data[idx]

def load_and_prepare_data(data, test_size=0.2):

    cond_tensor = data["cond"]
    target_tensor = data["target"]
    valid_time_array = data["valid_time"]

    # 過濾條件
    nonzero_ratio = (target_tensor != 0).float().mean(dim=(1, 2, 3))
    valid_idx = nonzero_ratio > 0.5
    cond_tensor = cond_tensor[valid_idx]
    target_tensor = target_tensor[valid_idx]
    valid_time_array = valid_time_array[valid_idx.numpy()]  # 注意要轉 numpy for indexing

    # === 依時間順序切分 ===
    N = len(cond_tensor)
    split_index = int(N * (1 - test_size))

    cond_train = cond_tensor[:split_index]
    cond_test = cond_tensor[split_index:]
    target_train = target_tensor[:split_index]
    target_test = target_tensor[split_index:]
    time_train = valid_time_array[:split_index]
    time_test = valid_time_array[split_index:]

    # 正規化參數（根據訓練集）
    cond_min = cond_train.min()
    cond_max = cond_train.max()
    target_min = target_train.min()
    target_max = target_train.max()

    # 套用正規化
    eps = 1e-6
    cond_train = (cond_train - cond_min) / (cond_max - cond_min + eps)
    cond_test = (cond_test - cond_min) / (cond_max - cond_min)
    target_train = (target_train - target_min) / (target_max - target_min)
    target_test = (target_test - target_min) / (target_max - target_min)


    # time_train → 對應訓練集每筆 target 的第一天時間
    # time_test → 對應測試集每筆 target 的第一天時間
    return cond_train, cond_test, target_train, target_test, cond_min, cond_max, target_min, target_max, time_train, time_test